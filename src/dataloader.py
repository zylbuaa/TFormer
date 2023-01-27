# -*- coding： utf-8 -*-
'''
@Time: 2022/5/11 20:44
@Author:YilanZhang
@Filename:dataloader.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''
from derm7pt.dataset import Derm7PtDatasetGroupInfrequent
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import torch
import cv2
import numpy as np
#Build the Pytorch dataloader
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,

)

aug = Compose(
    [   VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.8,rotate_limit=45,p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5),
        #RandomContrast(p=0.5),
        #RandomBrightness(p=0.5),
        # RandomGamma(p=0.5)
    ],
    p=0.5)

# dir_release = "/media/disk/zyl/data/derm7pt/release_v0/"
def load_dataset(dir_release):
    dir_meta = os.path.join(dir_release,'meta')
    dir_images = os.path.join(dir_release,'images')

    meta_df = pd.read_csv(os.path.join(dir_meta,'meta.csv'))
    train_indexes = list(pd.read_csv(os.path.join(dir_meta,'train_indexes.csv'))['indexes'])
    valid_indexes = list(pd.read_csv(os.path.join(dir_meta,'valid_indexes.csv'))['indexes'])
    test_indexes = list(pd.read_csv(os.path.join(dir_meta,'test_indexes.csv'))['indexes'])

    # The dataset after grouping infrequent labels
    derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images,
                                                    metadata_df=meta_df.copy(),
                                                    train_indexes=train_indexes,
                                                    valid_indexes=valid_indexes,
                                                    test_indexes=test_indexes)

    # Print the details of dataset
    derm_data_group.dataset_stats()

    return derm_data_group

train_data_transformation = transforms.Compose([
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #从头开始训练的均值和标准差
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet的均值和标准差
])

test_data_transformation = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #从头开始训练的均值和标准差
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet的均值和标准差
])

def load_image(path, shape):
    img = cv2.imread(path)
    img = cv2.resize(img, (shape[0], shape[1]))

    return img


# 做多分类的任务
class dataset(Dataset):
    def __init__(self,derm,shape,mode='train'):
        self.shape = shape
        self.mode = mode
        self.derm_paths = derm.get_img_paths(data_type=mode, img_type='derm')
        self.clinic_paths = derm.get_img_paths(data_type=mode, img_type='clinic')
        self.labels = derm.get_labels(data_type=mode,one_hot=False) #dict list
        if self.mode == 'train':
            self.meta = derm.meta_train.values
        elif self.mode == 'valid':
            self.meta = derm.meta_valid.values
        else:
            self.meta = derm.meta_test.values


    def __getitem__(self, index):
        # get the dermoscopy image path
        dermoscopy_img_path = self.derm_paths[index]
        # get the clinic image path
        clinic_img_path = self.clinic_paths[index]
        # load the dermoscopy image
        dermoscopy_img = load_image(dermoscopy_img_path,self.shape)
        # load the clinic image
        clinic_img = load_image(clinic_img_path,self.shape)

        if self.mode == 'train':
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255) # 归一化，转换为Tenor
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)

        # label
        DIAG = torch.LongTensor([self.labels['DIAG'][index]])

        # seven point check list
        # TODO
        PN = torch.LongTensor([self.labels['PN'][index]])
        BWV = torch.LongTensor([self.labels['BWV'][index]])
        VS = torch.LongTensor([self.labels['VS'][index]])
        PIG = torch.LongTensor([self.labels['PIG'][index]])
        STR = torch.LongTensor([self.labels['STR'][index]])
        DaG = torch.LongTensor([self.labels['DaG'][index]])
        RS = torch.LongTensor([self.labels['RS'][index]])


        #meta data
        metadata = torch.from_numpy(self.meta[index])

        return dermoscopy_img,clinic_img,metadata,[DIAG,PN,BWV,VS,PIG,STR,DaG,RS]

    def __len__(self):
        return len(self.clinic_paths)


if __name__ == '__main__':
    dir_release = "/media/disk/zyl/data/derm7pt/release_v0/"
    derm_data_group= load_dataset(dir_release)
    mata = derm_data_group.meta_train.values
    print(mata)
