import os
import torch
import torch.distributed as dist
from functools import partial


import argparse
from loguru import logger

import util.util as TheUtil
import data as TheData
import trainer as TheTrainer


def main(main_options):
    # set seed and cuDNN environment
    torch.backends.cudnn.enabled = True

    if main_options['speed_up']['ena_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info('TF32 enabled')
    logger.warning('CuDNN for acceleration enabled by setting torch.backends.cudnn.enabled as True')
    TheUtil.set_seed(main_options['seed'])
    logger.info(f"Seed set as {main_options['seed']}")
    if main_options['use_non_deterministic']:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        logger.warning('Use non deterministic algorithm')

    # set log file
    if main_options['local_rank'] <= 0:
        logger.add(main_options['path']['log_file'], level=main_options['log_level'])
    else:
        logger.remove()

    # set dataloader
    train_loaders_dict, valid_loaders_dict = TheData.create_dataloader(main_options)

    # set trainer
    trainer = TheTrainer.create_trainer(main_options, train_loaders_dict, valid_loaders_dict)

    # modify options based on different phase
    main_options['path']['profile_path'] = None

    if main_options['phase'] == 'debug':
        main_options['train']['val_iter'] = 20
        logger.critical('DEBUG MODE is enabled, val_iter is set to 20!')
    elif main_options['phase'] == 'profile':
        main_options['path']['profile_path'] = os.path.join(main_options['path']['tensorboard_log_dir'], 'profile')
        logger.critical('PROFILE MODE is enabled.')
    elif main_options['phase'] == 'detect':
        torch.autograd.set_detect_anomaly(True)

        def check_nan_hook(module_name, module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                for i, out in enumerate(output_tensor):
                    if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                        raise ValueError(
                            f"NaN detected in output {i} of module {module_name} ({module.__class__.__name__})")
            elif isinstance(output_tensor, torch.Tensor) and torch.isnan(output_tensor).any():
                raise ValueError(f"NaN detected in output of module {module_name} ({module.__class__.__name__})")

        for name, module in trainer.network.named_modules():
            if isinstance(module, torch.nn.Module):  # 确保是模块
                hook = partial(check_nan_hook, name)
                module.register_forward_hook(hook)

        logger.critical('DETECT MODE is enabled.')

    try:
        trainer.train(main_options)

    except Exception:
        logger.exception('Exception during training:')
        raise
    finally:
        if main_options['local_rank'] != -1:
            cleanup()
        logger.info('End Execution')

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    logger.info('Start Execution...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'debug', 'profile', 'detect'], help='Run train, debug, profile or detect', default='train')
    parser.add_argument('-l', '--local-rank', type=int, default=-1, help='node rank for distributed training')

    args = parser.parse_args()

    # DDP setting
    if args.local_rank == -1:
        os.environ.update({'LOCAL_RANK': '-1'})

    if args.local_rank != -1:
        # DDP training
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method="env://")

    options = TheUtil.parse_config_json(args)
    main(options)
