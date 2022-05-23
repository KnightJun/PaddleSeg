# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import sys
import cv2
import numpy as np
import random
import paddle
from paddleseg.cvlibs import manager
sys.path.append(r"D:\hqj\test\Pubilc\PaddleSeg\contrib\Matting")
import transforms as T
from BgsData import BgsData

@manager.DATASETS.add_component
class BgsubDataset(paddle.io.Dataset):
    """
    Pass in a dataset that conforms to the format.
        matting_dataset/
        |--bg/
        |
        |--train/
        |  |--fg/
        |  |--alpha/
        |
        |--val/
        |  |--fg/
        |  |--alpha/
        |  |--trimap/ (if existing)
        |
        |--train.txt
        |
        |--val.txt
    See README.md for more information of dataset.

    Args:
        dataset_root(str): The root path of dataset.
        transforms(list):  Transforms for image.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'trainval'). Default: 'train'.
        train_file (str|list, optional): File list is used to train. It should be `foreground_image.png background_image.png`
            or `foreground_image.png`. It shold be provided if mode equal to 'train'. Default: None.
        val_file (str|list, optional): File list is used to evaluation. It should be `foreground_image.png background_image.png`
            or `foreground_image.png` or ``foreground_image.png background_image.png trimap_image.png`.
            It shold be provided if mode equal to 'val'. Default: None.
        get_trimap (bool, optional): Whether to get triamp. Default: True.
        separator (str, optional): The separator of train_file or val_file. If file name contains ' ', '|' may be perfect. Default: ' '.
        key_del (tuple|list, optional): The key which is not need will be delete to accellect data reader. Default: None.
    """

    def __init__(self,
                 dataset_root,
                 transforms,
                 mode='train',
                 train_file=None,
                 val_file=None,
                 get_trimap=True,
                 separator=' ',
                 key_del=None,
                 batch_size=10):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = T.Compose(transforms)
        self.mode = mode
        self.get_trimap = get_trimap
        self.separator = separator
        self.key_del = key_del
        self.batch_size = batch_size

        if mode == 'trainval':
            raise ValueError(
                "unsupport trainval mode"
            )
        # check file
        if mode == 'train' or mode == 'trainval':
            if train_file is None:
                raise ValueError(
                    "When `mode` is 'train' or 'trainval', `train_file must be provided!"
                )
            self.train_data = BgsData()
            self.train_data.openData(os.path.join(dataset_root, train_file + ".meta"), os.path.join(dataset_root, train_file + ".data"))
            radiosLen = len(self.train_data.getRadios())
            self.train_data_lenlist = [ int(self.train_data.getRadiosDataSize(i) / batch_size) * batch_size for i in range(radiosLen) ]

        if mode == 'val' or mode == 'trainval':
            if val_file is None:
                raise ValueError(
                    "When `mode` is 'val' or 'trainval', `val_file must be provided!"
                )
            self.val_data = BgsData()
            self.val_data.openData(os.path.join(dataset_root, val_file + ".meta"), os.path.join(dataset_root, val_file + ".data"))
            radiosLen = len(self.val_data.getRadios())
            self.val_data_lenlist = [ int(self.val_data.getRadiosDataSize(i) / batch_size) * batch_size for i in range(radiosLen) ]

    def __getitem__(self, idx):
        radiusIdx = 0
        if self.mode == 'val':
            self.train_data = self.val_data
            self.train_data_lenlist = self.val_data_lenlist
        data = {}
        for v in self.train_data_lenlist:
            if idx < v:
                break
            idx -= v
            radiusIdx += 1
        
        fg, alpha = self.train_data.getImageData(radiusIdx, idx)
        data['alpha'] = alpha
        data['gt_fields'] = []

        # line is: fg [bg] [trimap]
        data['img'] = fg
        if self.mode in ['train', 'trainval']:
            data['fg'] = fg.copy()
            data['bg'] = fg.copy()
            data['gt_fields'].append('fg')
            data['gt_fields'].append('bg')
            data['gt_fields'].append('alpha')

        data['trans_info'] = []  # Record shape change information

        # Delete key which is not need
        if self.key_del is not None:
            for key in self.key_del:
                if key in data.keys():
                    data.pop(key)
                if key in data['gt_fields']:
                    data['gt_fields'].remove(key)
        data = self.transforms(data)

        # When evaluation, gt should not be transforms.
        if self.mode == 'val':
            data['gt_fields'].append('alpha')

        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')

        data['alpha'] = data['alpha'][np.newaxis, :, :] / 255.

        return data

    def __len__(self):
        if self.mode == 'train':
            return sum(self.train_data_lenlist)
        else:
            return sum(self.val_data_lenlist)

if __name__ == '__main__':
    def testT(data):
        return data
    ds = BgsubDataset(r"D:\hqj\test\private\ImageSpider", [testT], train_file='bgData', batch_size=3)
    print(len(ds))
    print(ds[39])