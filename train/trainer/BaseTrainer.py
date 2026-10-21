import random
import numpy as np
import torch
import tqdm
from loguru import logger
import os
from PIL import Image
import time
import torch.distributed as dist

import data
import util.util as TheUtil
import json
from torch.cuda.amp import autocast, GradScaler
from util.video_inference import predict_video
from util.checkpoint import load_checkpoint, load_model_weights
import csv
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from util.Profiler import Profiler
from util.scheduler import ReproducibleScheduler


class BaseTrainer:
    def __init__(self, networks, train_loaders_dict, valid_loaders_dict, losses, metrics, optimizer,
                 resume_state, init_method, tensorboard_log_dir, options):
        self.local_rank = options['local_rank']
        if options['local_rank'] == -1:
            self.network = networks['default'].cuda()
        else:
            network = networks['default'].cuda(options['local_rank'])
            find_unused_parameters = (options['train']['DDP_find_unused_parameters'] is not None and options['train']['DDP_find_unused_parameters'])
            self.network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[options['local_rank']], find_unused_parameters=find_unused_parameters)

        self.train_loader = data.InfinityIterator(train_loaders_dict['Paired'])
        self.valid_loaders_dict = valid_loaders_dict

        assert len(losses) > 0
        self.loss_function_dict = losses
        self.loss_recorder = {}
        self.clear_loss_recorder()

        assert options['train']['val_iter'] % options['train']['report_iter'] == 0
        assert options['train']['save_iter'] % options['train']['report_iter'] == 0

        self.metrics = metrics
        self.metric_csv_path = options['path']['metric_csv']
        self.init_csv()

        optimizer['args'].update({
            "params":
                list(filter(lambda p: p.requires_grad, self.network.parameters()))
        })
        self.optimizer = TheUtil.init_obj(optimizer, default_file_name='torch.optim', init_type='Optimizer')

        # lr scheduler
        self.scheduler = ReproducibleScheduler(
            optimizer=self.optimizer,
            strategy=options['train']['lr_scheduler'],
            last_epoch=options['train']['n_iter'] // options['train']['report_iter']
        )
        logger.success('LR scheduler is created in {} method.'.format(options['train']['lr_scheduler']))

        # resume training state
        if resume_state is None:
            self.global_step = 0

            self.init_weights(self.network, init_type=init_method)
        else:
            self.load_from_disk(resume_state, options)

        self.summery_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # init current_best_info
        self.current_best_info = {}
        for val_set_name in self.valid_loaders_dict:
            self.current_best_info[val_set_name] = {}
            for metric_name in self.metrics:
                self.current_best_info[val_set_name][metric_name] = {
                    'value': -2147483647,
                    'iter': -1
                }

        self.global_best_info = {
            'value': -2147483647,
            'iter': -1
        }

    def train(self, options):
        logger.info('Train start')
        if options['speed_up']['enable_amp']:
            logger.info('AMP enable.')
        else:
            logger.info('AMP disable.')

        # switch to train mode
        self.network.train()

        scaler = GradScaler(enabled=options['speed_up']['enable_amp'])
        torch.cuda.empty_cache()

        self.optimizer.zero_grad(set_to_none=True)

        with Profiler(options['path']['profile_path']) as prof:
            while self.global_step <= options['train']['n_iter']:
                logger.info('Train of iter {} start'.format(self.global_step + 1))
                logger.info('Current LearningRate {}'.format(self.scheduler.get_lr()))
                self.summery_writer.add_scalar('Learning rate', self.scheduler.get_lr(), self.global_step)

                if options['local_rank'] <= 0:
                    tqdm_bar = tqdm.tqdm(range(0, options['train']['report_iter']), desc='Report {}'.format(self.global_step // options['train']['report_iter']))
                else:
                    tqdm_bar = range(0, options['train']['report_iter'])

                if options['phase'] == 'debug':
                    # debug mode
                    for _ in tqdm_bar:
                        self.train_step(options, self.global_step, scaler)
                        self.global_step += 1

                        prof.step()

                        if self.global_step % options['train']['val_iter'] == 0 and self.global_step != 0:
                            # val part
                            logger.info('Validation of iter {} start in debug mode'.format(self.global_step))
                            self.eval_step(self.global_step, options)
                            logger.info('Validation of iter {} finish '.format(self.global_step))
                else:
                    for _ in tqdm_bar:
                        self.train_step(options, self.global_step, scaler)
                        self.global_step += 1

                        prof.step()
                
                logger.info('Train of iter {} finish'.format(self.global_step))
                self.print_loss_per_report_iter()

                # save last weights
                if self.local_rank <= 0:
                    self.save_everything('latest', self.global_step, options)
                    logger.success('Latest checkpoint saved.')

                if self.global_step % options['train']['val_iter'] == 0 and self.global_step != 0:
                    # val part
                    logger.info('Validation of iter {} start'.format(self.global_step))
                    self.eval_step(self.global_step, options)
                    logger.info('Validation of iter {} finish'.format(self.global_step))

                if self.global_step % options['train']['save_iter'] == 0 and self.global_step != 0 and self.local_rank <= 0:
                    self.save_everything(self.global_step, self.global_step, options)
                    logger.info('Checkpoint of iter {} saved'.format(self.global_step))

                if self.global_step >= options['train']['finish_iter']:
                    logger.info('Reach finish iter {}'.format(options['train']['finish_iter']))
                    break

                self.scheduler.adjust_lr(self.global_step // options['train']['report_iter'])

        logger.info('Train end')

    def train_step(self, options, iter_index, scaler):
        train_data = next(self.train_loader)

        frames = train_data['frames'].cuda()
        labels = train_data['labels'].cuda()

        with autocast(enabled=options['speed_up']['enable_amp']):
            output = self.network(frames)

            loss_supervised = self.loss_fn(
                output,
                labels.squeeze(1),
                'supervised_loss', iter_index,
                'Supervised Loss')

            loss = loss_supervised
            loss = loss / options['train']['iter_per_optim_step']

        # calc gradient and backward
        if options['speed_up']['enable_amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Grad clip
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=20, norm_type=2)

        if iter_index % options['train']['iter_per_optim_step'] == (options['train']['iter_per_optim_step'] - 1):
            if options['speed_up']['enable_amp']:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def eval_step(self, iter_index, options):
        self.network.eval()

        # # when changing the input size, the benchmark will re-performed, which would cause performance drop.
        # prev_benchmark_enable = torch.backends.cudnn.benchmark
        # torch.backends.cudnn.benchmark = False

        with torch.no_grad():
            eval_metric_result = {
                'iter': iter_index,
                'result': {}
            }
            for eval_set_name, eval_loader in self.valid_loaders_dict.items():
                eval_metric_result['result'][eval_set_name] = {}

                eval_loader_pbar = tqdm.tqdm(eval_loader)
                metrics_result = {}
                for metric_name in self.metrics:
                    metrics_result[metric_name] = {}

                for val_data in eval_loader_pbar:
                    video_path = val_data['video_path']
                    frames = val_data['frames'].cuda()
                    labels = val_data['labels'].cuda()
                    eval_batch_size = frames.shape[0]
                    # print(frames.shape)

                    assert eval_batch_size == 1

                    with autocast(enabled=options['speed_up']['fast_eval']):
                        result_is_T1 = bool(predict_video(self.network, frames))
                    
                    # with autocast(device_type='cuda', enabled=options['speed_up']['fast_eval']):
                    #     output = self.network(
                    #         frames[:, :, -16:]
                    #     )
                    #     output = output[0]
                    #     if output[0] < output[1]:
                    #         result_is_T1 = True

                    with autocast(enabled=options['speed_up']['fast_eval']):
                        for metric_name, metric in self.metrics.items():
                            metric_result = metric(result_is_T1, labels[0])
                            for sample_i in range(eval_batch_size):
                                metrics_result[metric_name][video_path[sample_i]] = metric_result

                eval_metric_result['result'][eval_set_name] = metrics_result

            # log & visualization
            self.metric_result_log_and_visual(eval_metric_result)

            # save best checkpoint
            self.check_and_save_best_checkpoint(iter_index, eval_metric_result, options)

        self.network.train()

        # torch.backends.cudnn.benchmark = prev_benchmark_enable

    def loss_fn(self, x, y, loss_name, iter_index, tag):
        loss_sum = 0
        for loss in self.loss_function_dict[loss_name]:
            single_loss = loss(x, y)
            loss_sum = loss_sum + single_loss * loss.weight_from_iter(iter_index)

            with torch.no_grad():
                # visualization
                self.summery_writer.add_scalar(
                    '[{}@{}] {}'.format(tag, self.local_rank, loss.__class__.__name__), single_loss.item(),
                    self.global_step)

        with torch.no_grad():
            if not isinstance(loss_sum, type(0)):
                loss_sum_item = loss_sum.item()
                self.summery_writer.add_scalar('[{}@{}] Total Loss'.format(tag, self.local_rank), loss_sum_item, self.global_step)
                self.loss_recorder[loss_name].append(loss_sum_item)

        return loss_sum

    # Tool Functions
    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    logger.critical(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                    raise NotImplementedError(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        if init_type is None:
            logger.info('{} initialized as default'.format(net.__class__.__name__))
        else:
            net.apply(init_func)
            logger.success("{} initialized as {}".format(net.__class__.__name__, init_type))

    def print_loss_per_report_iter(self):
        logger.info('Current Loss:')
        for loss_name, loss_value in self.loss_recorder.items():
            if len(loss_value) > 0:
                logger.info('\t{}: {}'.format(loss_name, np.mean(loss_value)))

        self.clear_loss_recorder()

    def clear_loss_recorder(self):
        for loss_name, loss_funcs in self.loss_function_dict.items():
            if len(loss_funcs) > 0:
                self.loss_recorder[loss_name] = []

    def save_everything(self, prefix, iter_index, options):
        if self.local_rank == -1:
            network_label = self.network.__class__.__name__
            save_network(prefix, network=self.network, network_label=network_label,
                         save_dir=options['path']['checkpoint'])
        elif self.local_rank == 0:
            network_label = self.network.module.__class__.__name__
            save_network(prefix, network=self.network.module, network_label=network_label,
                         save_dir=options['path']['checkpoint'])
        else:
            # other process is no need to save anything
            return

        # Train info
        train_info = {
            'iter': iter_index,

            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.get_state(),

            # random generator save
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate()
        }
        torch.save(train_info, os.path.join(options['path']['checkpoint'], "{}_train_info.pth".format(prefix)))
        logger.success('{} saved'.format(prefix))

    def load_from_disk(self, resume_state, options):
        if options['local_rank'] != -1:
            network_label = self.network.module.__class__.__name__
            load_network(resume_state, network=self.network.module, network_label=network_label)
        else:
            network_label = self.network.__class__.__name__
            load_network(resume_state, network=self.network, network_label=network_label)

        # Train info
        train_info = load_checkpoint(resume_state + '_train_info.pth',
                                     trust_checkpoint=bool(options['path']['trust_resume_state']))
        self.global_step = train_info['iter']

        self.optimizer.load_state_dict(train_info['optimizer'])
        self.scheduler.set_state(train_info['lr_scheduler'])

        if options['local_rank'] == -1:
            # RNG should not be loaded in the DDP mode
            torch.set_rng_state(train_info['torch_rng'])
            torch.cuda.set_rng_state(train_info['torch_cuda_rng'])
            np.random.set_state(train_info['numpy_rng'])
            random.setstate(train_info['python_rng'])

    def init_csv(self):
        csv_header = 'iter,'
        for val_set_name in self.valid_loaders_dict:
            for metric in self.metrics:
                csv_header += '{}_{},'.format(val_set_name, metric)

        csv_header = csv_header[:-1]
        with open(self.metric_csv_path, 'w') as csv_file:
            csv_file.write(csv_header)

    def metric_result_log_and_visual(self, metric_result_dict):
        logger.info('Eval result at iter {}:'.format(metric_result_dict['iter']))

        csv_line = '\n{},'.format(metric_result_dict['iter'])
        for dataset_name, dataset_result in metric_result_dict['result'].items():
            for metric_name, metric_dict in dataset_result.items():
                avg_metric_value = np.mean(list(metric_dict.values()))

                self.summery_writer.add_scalar('[{}] {}'.format(dataset_name, metric_name),
                                               avg_metric_value, metric_result_dict['iter'])
                csv_line += str(float(avg_metric_value)) + ','

                logger.info('\t{} at {}: {:.6f}'.format(metric_name, dataset_name, avg_metric_value))
                # logger_str = f'Detail {metric_name}:\n'
                # for sample_name, sample_value in metric_dict.items():
                #     logger_str += f'\t{sample_name}: {sample_value:.6f}'

                # logger.info(logger_str)

        csv_line = csv_line[:-1]
        with open(self.metric_csv_path, 'a+') as csv_file:
            csv_file.write(csv_line)

    def check_and_save_best_checkpoint(self, iter_index, eval_metric_result, options):
        watching_metrics = options['trainer']['watching_metrics']

        metric_sum = 0
        # discover best
        for val_set_name, val_result_info in self.current_best_info.items():
            for metric_name, metric_result_info in val_result_info.items():
                val_set_metric_name = '{}_{}'.format(val_set_name, metric_name)
                if val_set_metric_name not in watching_metrics:
                    continue

                metric_sum += np.mean(list(eval_metric_result['result'][val_set_name][metric_name].values()))
                single_better = np.mean(list(eval_metric_result['result'][val_set_name][metric_name].values()))

                if metric_result_info['value'] < single_better:
                    # new best
                    logger.info('New Best {} under {} : {}'.format(metric_name, val_set_name, single_better))
                    self.current_best_info[val_set_name][metric_name]['value'] = float(single_better)
                    self.current_best_info[val_set_name][metric_name]['iter'] = iter_index

                    # save network
                    self.save_everything(val_set_metric_name, iter_index, options)

        if self.global_best_info['value'] < metric_sum:
            self.global_best_info['value'] = float(metric_sum)
            self.global_best_info['iter'] = iter_index

            logger.info('New Global Best : {}'.format(metric_sum))
            # save network
            self.save_everything('global', iter_index, options)

def load_network(resume_state, network, network_label):
    if resume_state is None:
        return
    logger.info('Begin loading pretrained model [{:s}] ...'.format(network_label))

    model_path = "{}_{}.pth".format(resume_state, network_label)

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))

    load_model_weights(network, model_path, strict=True)

    logger.success('Load pretrained model {} success.'.format(network_label))


def save_network(prefix, network, network_label, save_dir):
    """ save network structure """
    save_filename = '{}_{}.pth'.format(prefix, network_label)
    save_path = os.path.join(save_dir, save_filename)

    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
