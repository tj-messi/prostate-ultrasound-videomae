# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib.pyplot as plt
import numpy as np
import utils
from util.Profiler import Profiler

from piq import ssim, SSIMLoss ,psnr
import os

profile_path = None

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    criterion_ploss : torch.nn.Module,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    export_reconstructions=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    profile_path = None
    with Profiler(profile_path) as prof:
        for step, batch in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step  # global training iteration
            # 取消lr缩放 
            # lr_schedule_values = None
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group[
                            "lr_scale"]
                        # lr_record = param_group["lr"]
                    if wd_schedule_values is not None and param_group[
                            "weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            # NOTE: When the decoder mask ratio is 0,
            # in other words, when decoder masking is not used,
            # decode_masked_pos = ~bool_masked_pos
            images, bool_masked_pos, decode_masked_pos = batch

            images = images.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(
                device, non_blocking=True).flatten(1).to(torch.bool)
            decode_masked_pos = decode_masked_pos.to(
                device, non_blocking=True).flatten(1).to(torch.bool)

            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                        None,
                                                                        None,
                                                                        None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                    None,
                                                                    None,
                                                                    None]
                unnorm_images = images * std + mean  # in [0, 1]

                if normlize_target:
                    images_squeeze = rearrange(
                        unnorm_images,
                        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                        p0=2,
                        p1=patch_size,
                        p2=patch_size)
                    images_norm = (images_squeeze - images_squeeze.mean(
                        dim=-2, keepdim=True)) / (
                            images_squeeze.var(
                                dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    images_patch = rearrange(
                        unnorm_images,
                        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                        p0=2,
                        p1=patch_size,
                        p2=patch_size)

                B, N, C = images_patch.shape
                labels = images_patch[~decode_masked_pos].reshape(B, -1, C) # images_patch torch.Size([16, 1568, 1536])

            if loss_scaler is None:
                # outputs = model(images, bool_masked_pos, decode_masked_pos)
                # loss = (outputs - labels)**2
                # loss = loss.mean(dim=-1)
                # cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
                # loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()
                outputs = model(images, bool_masked_pos, decode_masked_pos)
                # true label [ vis_patch(B, N_vis, 1536) +  mask_patch(B, N_mask, 1536) ] 
                vis_patch = images_patch[~bool_masked_pos].reshape(B, -1, C)
                mask_patch = images_patch[bool_masked_pos].reshape(B, -1, C)
                true_label = torch.cat([vis_patch, mask_patch], dim=1)
                
                # loss = (outputs - true_label) ** 2
                # loss = torch.mean(loss)
                
                vis_outputs = outputs[:, :vis_patch.shape[1], :]  # 获取可见部分的输出
                mask_outputs = outputs[:, vis_patch.shape[1]:, :]  # 获取遮掩部分的输出
                vis_loss = (vis_outputs - vis_patch) ** 2
                vis_loss = torch.mean(vis_loss) * 0.0
                mask_loss = (mask_outputs - mask_patch) ** 2
                mask_loss = torch.mean(mask_loss)
                mask_loss = mask_loss * 1.0 
                # 加权和
                loss = vis_loss + mask_loss
                
            else: 
                with torch.cuda.amp.autocast():
                    print("amp used")
                    outputs = model(images, bool_masked_pos, decode_masked_pos)
                    # true label [ vis_patch(B, N_vis, 1536) +  mask_patch(B, N_mask, 1536) ] 
                    vis_patch = images_patch[~bool_masked_pos].reshape(B, -1, C)
                    mask_patch = images_patch[bool_masked_pos].reshape(B, -1, C)
                    true_label = torch.cat([vis_patch, mask_patch], dim=1)
                    # loss = (outputs - true_label)**2
                    # loss = torch.mean(loss)
                    
                    vis_outputs = outputs[:, :vis_patch.shape[1], :]  # 获取可见部分的输出
                    mask_outputs = outputs[:, vis_patch.shape[1]:, :]  # 获取遮掩部分的输出
                    vis_loss = (vis_outputs - vis_patch) ** 2
                    vis_loss = torch.mean(vis_loss)
                    mask_loss = (mask_outputs - mask_patch) ** 2
                    mask_loss = torch.mean(mask_loss)
                    mask_loss = mask_loss * 10
                    # 加权和
                    loss = vis_loss + mask_loss

            '''
            ssim loss
            '''

            if export_reconstructions and epoch % 5 == 4 and step == 5:
                # 计算perceptual loss
                restored_outputs = torch.zeros_like(outputs)
                # 对每个样本进行处理
                for b in range(B):
                    # bool_masked_pos，提取可见和遮掩的索引
                    masked_indices = bool_masked_pos[b].nonzero(as_tuple=True)[0]  # 获取 mask patch 的索引
                    visible_indices = (~bool_masked_pos[b]).nonzero(as_tuple=True)[0]  # 获取 vis patch 的索引

                    # 将 outputs[b] 中的前面 patches（按 visible_indices）恢复到 restored_outputs[b] 中的指定位置
                    for i, visible_idx in enumerate(visible_indices):  # 遍历 visible_indices 和对应的位置
                        restored_outputs[b, visible_idx] = outputs[b, i]  # 将 outputs[b, i] 填充到 restored_outputs[b, visible_idx]

                    # 对于 masked_indices 可以继续类似操作，如果需要填充其他部分
                    for i, masked_idx in enumerate(masked_indices):  # 遍历 masked_indices 和对应的位置
                        restored_outputs[b, masked_idx] = outputs[b, len(visible_indices) + i]  # 填充 mask_patch
                        
                restored_labels = torch.zeros_like(true_label)
                # 对每个样本进行处理
                for b in range(B):
                    masked_indices = bool_masked_pos[b].nonzero(as_tuple=True)[0]  # 获取 mask patch 的索引
                    visible_indices = (~bool_masked_pos[b]).nonzero(as_tuple=True)[0]  # 获取 vis patch 的索引
                    
                    # 将 true_label[b] 中的前面 patches（按 visible_indices）恢复到 restored_outputs[b] 中的指定位置
                    for i, visible_idx in enumerate(visible_indices):  # 遍历 visible_indices 和对应的位置
                        restored_labels[b, visible_idx] = true_label[b, i]  # 将 true_label[b, i] 填充到 restored_labels[b, visible_idx]

                    # 对于 masked_indices 可以继续类似操作，如果需要填充其他部分
                    for i, masked_idx in enumerate(masked_indices):  # 遍历 masked_indices 和对应的位置
                        restored_labels[b, masked_idx] = true_label[b, len(visible_indices) + i]  # 填充 mask_patch
                        
                output_images = rearrange(
                            restored_outputs,
                            'b (t h w) (p0 p1 p2 c) -> b (t h w) (p0 p1 p2) c',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            )
                
                output_images = (output_images ) * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                output_images = rearrange(output_images, 'b n p c -> b n (p c)')
                output_images = rearrange(
                            output_images,
                            'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            )
                
                label_images = rearrange(
                            restored_labels,
                            'b (t h w) (p0 p1 p2 c) -> b (t h w) (p0 p1 p2) c',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            )
                
                label_images = (label_images ) * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                label_images = rearrange(label_images, 'b n p c -> b n (p c)')
                label_images = rearrange(
                            label_images,
                            'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            )
                
                
                # # [ B , C , T , H , W ]
                # # for b in range(B):
                # #     for t in range(images.shape[2]):
                # #         output_image = output_images[b, :, t, :, :].unsqueeze(0).float()
                # #         label_image = label_images[b, :, t, :, :].unsqueeze(0).float()
                # #         ploss = criterion_ploss(output_image, label_image) *25
                # #         # ploss_item = ploss.item()
                # #         total_ploss += ploss
                        
                # # avg_ploss = total_ploss / (B * images.shape[2])
                # # [B*T, C, H, W]
                output_frames = output_images.permute(0, 2, 1, 3, 4).reshape(B * images.shape[2], 3, 224, 224).float()
                label_frames = label_images.permute(0, 2, 1, 3, 4).reshape(B * images.shape[2], 3, 224, 224).float()
                
                # #loss = torch.mean((output_frames - label_frames)**2, dim=(0, 1, 2, 3)) 
                # loss = torch.mean((outputs - true_label)**2) 
            

            # # 一次性计算所有帧的ssim损失
            # if epoch % 10 == 9 and step == 5:
                output_frames_ssim = output_frames.detach().clamp(0,1)
                label_frames_ssim = label_frames.detach().clamp(0,1)
                # ssimloss = criterion_ploss(output_frames_ssim, label_frames_ssim)
                psnr_value = psnr(output_frames_ssim, label_frames_ssim, data_range=1.)
                # # ssimloss = 0  
                del output_frames_ssim, label_frames_ssim
                print(f'============ Epoch: {epoch}\tStep: {step}\tPSNR: {psnr_value:.4f}')
            
            # # 计算平均损失
            # avg_ploss = ssimloss  
            
            if export_reconstructions and epoch % 5 == 4 and step == 5:
                '''
                可视化（正确）
                '''

                # 针对output 可视化 [vis_patch(B, N_vis, 1536) +  mask_patch(B, N_mask, 1536)] 
                # 需要根据 bool masked pos 去还原原来的 patch output [B,N,C] 再rearrange到  [B,T,C,H,W]
                
                restored_outputs = torch.zeros_like(outputs)
                # 对每个样本进行处理
                for b in range(B):
                    # bool_masked_pos，提取可见和遮掩的索引
                    masked_indices = bool_masked_pos[b].nonzero(as_tuple=True)[0]  # 获取 mask patch 的索引
                    visible_indices = (~bool_masked_pos[b]).nonzero(as_tuple=True)[0]  # 获取 vis patch 的索引

                    # 将 outputs[b] 中的前面 patches（按 visible_indices）恢复到 restored_outputs[b] 中的指定位置
                    for i, visible_idx in enumerate(visible_indices):  # 遍历 visible_indices 和对应的位置
                        restored_outputs[b, visible_idx] = outputs[b, i]  # 将 outputs[b, i] 填充到 restored_outputs[b, visible_idx]

                    # 对于 masked_indices 可以继续类似操作，如果需要填充其他部分
                    for i, masked_idx in enumerate(masked_indices):  # 遍历 masked_indices 和对应的位置
                        restored_outputs[b, masked_idx] = outputs[b, len(visible_indices) + i]  # 填充 mask_patch


                # 对 unnorm_images 进行 rearrange
                output_images = rearrange(
                            restored_outputs,
                            'b (t h w) (p0 p1 p2 c) -> b (t h w) (p0 p1 p2) c',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            )
                
                output_images = (output_images ) * (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + images_squeeze.mean(dim=-2, keepdim=True)
                output_images = rearrange(output_images, 'b n p c -> b n (p c)')
                output_images = rearrange(
                            output_images,
                            'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
                            t=images.shape[2] // 2,
                            h=images.shape[3] // patch_size,
                            w=images.shape[4] // patch_size,
                            p0=2,
                            p1=patch_size,
                            p2=patch_size,
                            c=images.shape[1]
                            ).clamp(0,1)
                # output_images = output_images * std + mean
                output_images = output_images * 255  # 将图像乘以 255

                for i in range(output_images.shape[0]):  # 对每个 B 进行处理
                    
                    output_image = output_images[i].cpu().detach().numpy().astype(np.uint8)  # 转换为 numpy 数组

                    C, T, H, W = output_image.shape[0], output_image.shape[1], output_image.shape[2], output_image.shape[3]  

                    # 将 16 个 patch 按 4 行 4 列拼接成一个大的图像
                    # 16 patches → 4x4 网格
                    grid_image = np.zeros((H * 4, W * 4, C), dtype=np.uint8)  # 4x4 网格，C是通道数

                    for j in range(T):  # 这里T是16
                        row = j // 4  # 计算当前 patch 应该放在哪一行
                        col = j % 4   # 计算当前 patch 应该放在哪一列

                        # 将每个 patch 放入对应的位置
                        grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = np.transpose(output_image[:,j], (1, 2, 0))

                    # 可视化和保存图像
                    fig, ax = plt.subplots(figsize=(12, 12))  # 选择合适的大小
                    ax.imshow(grid_image)
                    ax.axis('off')  # 隐藏坐标轴

                    # 保存可视化图像到文件
                    os.makedirs('./output/visualization', exist_ok=True)
                    plt.savefig(f'./output/visualization/output_patch_image_batch_{epoch}_{i}.png', bbox_inches='tight')
                    plt.close()  # 关闭当前的 figure，防止内存过度消耗
                
                unnorm_images_255 = unnorm_images * 255
                
                for i in range(images.shape[0]):  # 对每个 B 进行处理
                    # 获取每个图像的输出
                    image = unnorm_images_255[i].cpu().detach().numpy().astype(np.uint8) # 转换为 numpy 数组

                    C, T, H, W = image.shape[0], image.shape[1], image.shape[2], image.shape[3]  # 获取通道数、patch 数量和每个 patch 的尺寸

                    # 将 16 个 patch 按 4 行 4 列拼接成一个大的图像
                    # 16 patches → 4x4 网格
                    grid_image = np.zeros((H * 4, W * 4, C), dtype=np.uint8)  # 4x4 网格，C是通道数

                    for j in range(T):  # 这里T是16
                        row = j // 4  # 计算当前 patch 应该放在哪一行
                        col = j % 4   # 计算当前 patch 应该放在哪一列

                        # 将每个 patch 放入对应的位置
                        grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = np.transpose(image[:, j], (1, 2, 0))

                    # 可视化和保存图像
                    fig, ax = plt.subplots(figsize=(12, 12))  # 选择合适的大小
                    ax.imshow(grid_image)
                    ax.axis('off')  # 隐藏坐标轴

                    # 保存可视化图像到文件
                    os.makedirs('./output/visualization', exist_ok=True)
                    plt.savefig(f'./output/visualization/input_patch_image_batch_{epoch}_{i}.png', bbox_inches='tight')
                    plt.close()  # 关闭当前的 figure，防止内存过度消耗

            # loss_value = loss.item() 
            loss_value = loss.item()
            if step % 50 == 0:
                # print(f'============ Epoch: {epoch}\tStep: {step}\tMSE: {loss.item():.6f}\tssimLoss: {1}')
                print(f'============ Epoch: {epoch}\tStep: {step}\tMSE_loss: {loss.item():.6f}\tVis_loss: {vis_loss.item():.6f}\tMask_loss: {mask_loss.item():.6f}')
                
            loss = loss 

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(2)

            optimizer.zero_grad()

            if loss_scaler is None:
                loss.backward()
                if max_norm is None:
                    grad_norm = utils.get_grad_norm_(model.parameters())
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm)
                optimizer.step()
                loss_scale_value = 0
            else:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(
                    optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=is_second_order)
                loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr , head="opt")
                log_writer.update(min_lr=min_lr , head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)
                
            prof.step()
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # for i, param_group in enumerate(optimizer.param_groups):
    #     lr_record = param_group["lr"]  # DEBUG
    #     break
    # print(f'DEBUG\tEpoch: {epoch}\tLR: {lr_record:.6f}')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
