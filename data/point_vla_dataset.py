import os
import logging
import json
import yaml
import random

import numpy as np
import cv2

from torch.utils.data import Dataset

class AffordVLADataset(Dataset):
    def __init__(self,split='train'):
        self.DATASET_NAME = "hoi4d_points"

        # Load the config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'base.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.split = split

        self.points_path = "/home/panwen.hu/workspace/jian.zhang/EAI/datasets/HOI4D_points/"
        data_root='/home/panwen.hu/workspace/jian.zhang/EAI/datasets/HOI4D_KPST'
        self.origin_path = '/home/panwen.hu/workspace/jian.zhang/EAI/datasets/HOI4D_release'

        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root path not found: {data_root}")
        if not os.path.exists(self.origin_path):
            raise FileNotFoundError(f"Origin path not found: {self.origin_path}")

        with open(os.path.join(data_root, 'metadata.json'), "r") as fp:
            self.metadata = json.load(fp)[split]

        self.data_num = len(self.metadata)
        logging.info("Totally {} samples in {} set.".format(self.data_num, split))

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def __len__(self):
        return self.data_num

    def history_image(self):
        pass
    
    def get_item(self, index: int=None, state_only=False):
        return self.__getitem__(index)
    
    def __getitem__(self, idx: int=None):
        if idx is None:
            idx = random.randint(0, self.data_num-1)
        elif idx > self.data_num:
            raise ValueError(f"idx must be less than n_data={self.data_num}, but idx = {idx}")

        instruction = self.metadata[idx]['action'] + ' ' + self.metadata[idx]['object']
        data_fp_pure = '_'.join(self.metadata[idx]['index'].split(' '))
        img_idx = self.metadata[idx]['img']

        if 'fold_Clothes' in data_fp_pure:
            points_fp = os.path.join(self.points_path, self.split, data_fp_pure.split('/')[-1], str(img_idx),'points2d_center.npy')
            points_maxoffset_fp = os.path.join(self.points_path, self.split, data_fp_pure.split('/')[-1],str(img_idx),'points2d_max_offset.npy')
        else:
            points_fp = os.path.join(self.points_path, self.split, data_fp_pure, str(img_idx).zfill(5), 'points2d_center.npy')
            points_maxoffset_fp = os.path.join(self.points_path, self.split, data_fp_pure, str(img_idx).zfill(5), 'points2d_max_offset.npy')

        dtraj2d = np.load(points_fp)
        dtraj2d_maxoffset = np.load(points_maxoffset_fp)

        center_points_2d =  np.full((64, 4, 2), np.nan)
        max_offset_points_2d =  np.full((64, 4, 2), np.nan)

        center_points_2d[1,:,:] =  dtraj2d
        max_offset_points_2d[1,:,:] = dtraj2d_maxoffset
        
        imgs = []
        for img_i in range(max(img_idx - self.IMG_HISORY_SIZE+1, 0), img_idx+1):
            if 'fold_Clothes' in data_fp_pure:
                rgb_fp = os.path.join(self.origin_path, data_fp_pure, 'rgb_'+str(img_i)+'.png')
            else:
                rgb_fp = os.path.join(self.origin_path, data_fp_pure, 'align_rgb', str(img_i).zfill(5)+'.jpg')

            img = cv2.imread(rgb_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = np.stack(imgs)
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)
 
        images = imgs # IMG_HISORY_SIZE=2帧图片

        # 适配RDT数据格式
        cam_high_mask = np.array([True] * self.IMG_HISORY_SIZE)

        state = np.zeros((self.CHUNK_SIZE,self.STATE_DIM))
        state_indicator = np.zeros(self.STATE_DIM)
        actions = np.zeros((self.CHUNK_SIZE, self.STATE_DIM)) 

        cam_left_wrist = np.zeros_like(images)
        cam_right_wrist = np.zeros_like(images)
        cam_left_wrist_mask = np.array([False] * self.IMG_HISORY_SIZE)
        cam_right_wrist_mask = np.array([False] * self.IMG_HISORY_SIZE)

        meta = {
            "dataset_name": self.DATASET_NAME,
            'data_path':data_fp_pure,
            'image_index':img_idx,

            "instruction": instruction,
            "step_id": img_idx,
        }

        item ={
            'meta':meta,
            "state": state,
            "state_std": state,
            "state_mean": state,
            "state_norm": state,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask,

            'cam_high':images,
            "cam_high_mask": cam_high_mask,

            'points':{
                'center_points_2d':center_points_2d,
                'max_offset_points_2d':max_offset_points_2d
            },
        }

        return item

# for test the dataset
def test_dataset():
    from torch.utils.data import DataLoader
    import logging

    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化数据集
    dataset = AffordVLADataset(split='train')
    
    # 打印数据集大小
    print(f"Dataset size: {len(dataset)}")

    # 测试单个样本加载
    try:
        sample = dataset[0]
        print("Sample loaded successfully!")
        print(f"Instruction: {sample['meta']['instruction']}")
        print(f"Meta: {sample['meta']}")
        print(f"Center Points Shape: {sample['points']['center_points_2d'].shape}")
        print(f"Max Offset Points Shape: {sample['points']['max_offset_points_2d'].shape}")
        print(f"Camera (Image) Shape: {sample['cam_high'].shape}")
    except Exception as e:
        print(f"Error loading a sample: {e}")

    # 测试与 DataLoader 的兼容性
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    try:
        for i, batch in enumerate(dataloader):
            print(f"Batch {i+1} loaded successfully!")
            print(f"Batch Instructions: {batch['meta']['instruction']}")
            print(f"Batch Center Points Shape: {batch['points']['center_points_2d'].shape}")
            print(f"Batch Camera Shape: {batch['cam_high'].shape}")
            break  # 只测试第一个批次
    except Exception as e:
        print(f"Error loading a batch: {e}")


if __name__ == "__main__":
    test_dataset()