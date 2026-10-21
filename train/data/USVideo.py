import torch.utils.data as data
import os
import random
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation
from loguru import logger


class USVideoDataset(data.Dataset):
    def __init__(self, root_dir, patch_size=224, frame_count=16, full_seq=True, data_augmentation=False, gray_scale=False):
        super().__init__()

        t0_dir = os.path.join(root_dir, 'T0')
        t1_dir = os.path.join(root_dir, 'T1')

        t0_video_list = [n for n in os.listdir(t0_dir) if not n.startswith('.') and os.path.isdir(os.path.join(t0_dir, n))]
        t0_video_list.sort()

        t1_video_list = [n for n in os.listdir(t1_dir) if not n.startswith('.') and os.path.isdir(os.path.join(t1_dir, n))]
        t1_video_list.sort()

        self.video_list = []
        for t0_video in t0_video_list:
            self.video_list.append([
                os.path.join(t0_dir, t0_video), False
            ])

        for t1_video in t1_video_list:
            self.video_list.append([
                os.path.join(t1_dir, t1_video), True
            ])

        self.da_enable = data_augmentation
        if not self.video_list:
            raise ValueError('Dataset contains no video directories')
        if patch_size <= 0 or (not full_seq and frame_count <= 0):
            raise ValueError('patch_size and training frame_count must be positive')
        self.full_seq = full_seq
        self.frame_count = frame_count
        self.gray_scale = gray_scale

        self.to_tensor_transform = Compose([
            Resize((patch_size, patch_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        video_path, label = self.video_list[index]
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        frame_list = [n for n in os.listdir(video_path) if not n.startswith('.') and os.path.isfile(os.path.join(video_path, n)) and os.path.splitext(n)[1].lower() in extensions]
        if not frame_list:
            raise ValueError('Video contains no supported image frames: ' + video_path)
        frame_list.sort()
        frame_tensor_list = []

        if not self.full_seq:
            start_idx = random.randint(0, max(0, len(frame_list) - self.frame_count))
            end_idx = start_idx + self.frame_count
        else:
            start_idx = 0
            end_idx = len(frame_list)

        for frame_name in frame_list[start_idx:end_idx]:
            frame_path = os.path.join(video_path, frame_name)
            with Image.open(frame_path) as image:
                frame = image.convert('RGB')
            frame = self.to_tensor_transform(frame)
            frame_tensor_list.append(frame)

        cur_frame_tensor_list_len = len(frame_tensor_list)
        target_length = 16 if self.full_seq else self.frame_count
        for extend_idx in range(cur_frame_tensor_list_len, target_length):
            frame_tensor_list.append(frame)

        total_frame = torch.stack(frame_tensor_list, dim=0)

        if self.da_enable:
            # Apply the same spatial and temporal flips to every frame.
            hor_flip = random.choice([True, False])
            ver_flip = random.choice([True, False])
            tem_flip = random.choice([True, False])

            flip_dim_list = []
            if tem_flip:
                flip_dim_list.append(0)
            if ver_flip:
                flip_dim_list.append(2)
            if hor_flip:
                flip_dim_list.append(3)

            if len(flip_dim_list) > 0:
                total_frame = torch.flip(total_frame, dims=flip_dim_list)

        if label == True:
            label_tensor = torch.ones(1).long()
        else:
            label_tensor = torch.zeros(1).long()

        if self.gray_scale:
            gray_tensor = (total_frame[:, 0:1] + total_frame[:, 1:2] + total_frame[:, 2:3]) / 3
            total_frame = torch.concat([gray_tensor, gray_tensor, gray_tensor], dim=1)

        return {
            'video_path': video_path,
            'start_index': start_idx,
            'end_index': end_idx,
            'frames': total_frame.permute(1, 0, 2, 3),
            'labels': label_tensor
        }

    def __len__(self):
        return len(self.video_list)

