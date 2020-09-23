import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import os.path
from os import makedirs
import cv2
import matplotlib.pyplot as plt
import time
import zipfile

# cocoapi
from pycocotools.coco import COCO

import augmentations as aug
import constants as cons

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class CocoDataset(Dataset):
    """COCO Dataset for segmentation"""

    def __init__(self,train_data_path,valid_data_path,ann_path,valid=False,seed=0,transform=None,as_img=False):
        if valid:
            self.data_path = valid_data_path
            self.data_type = 'val2017'
        else:
            self.data_path = train_data_path
            self.data_type = 'train2017'
        self.transform = transform
        self.as_img = as_img
        self.coco = COCO(os.path.join(ann_path,'instances_'+self.data_type+'.json'))
        self.ids = list(self.coco.imgToAnns.keys())
        random.seed(seed)
        random.shuffle(self.ids)
        self.imgs = self.coco.loadImgs(self.ids)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = np.asarray([cat['id'] for cat in cats])
        
    def __len__(self):
        return len(self.imgs)
    
    def categoryIdToLabel(self,catId):
        """ return Label corresponding to Category Id."""
        return np.where(self.catIds == catId)[0][0]

    def __getitem__(self,idx):
        img = cv2.imread(os.path.join(self.data_path,self.data_type,self.imgs[idx]['file_name']))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.ids[idx]))        
        mask = np.zeros((img.shape[0],img.shape[1],cons.NUM_CLASSES), dtype=np.uint8)
        for ann in anns:
            mask[:,:,self.categoryIdToLabel(ann['category_id'])] |= self.coco.annToMask(ann)
        if self.transform is not None:
            augmented = self.transform(image=img,mask=mask)
            img, mask = augmented['image'], augmented['mask']
        if not self.as_img:
            img = img.transpose(2,0,1)
        mask = mask.transpose(2,0,1)
        return img, mask
    
def load_data(train_data_path,valid_data_path,ann_path,batch_size,shuffle=True,num_worker=0,seed=1,valid=False,transform=None,as_img=False):
    """generate Loader for COCO Dataset."""
    np.random.seed(seed)
    dataset = CocoDataset
    return DataLoader(
        dataset(train_data_path,valid_data_path,ann_path,seed=seed,valid=valid,transform=transform,as_img=as_img),
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers=num_worker,
        worker_init_fn=worker_init_fn)

def main():

    '''
    dataset=CocoDataset(
        data_path='coco',
        valid=False,
        transform=aug.transform_train,
        as_img=False)
    img, mask = dataset.__getitem__(1)
    '''

    loader = load_data(
        train_data_path='coco/images',
        valid_data_path='coco/images',
        ann_path='coco/annotations',
        batch_size=16,
        num_worker=0,
        valid=False,
        transform=aug.transform_train,
        as_img=True)

    imgs, masks = next(loader.__iter__())
    '''
    for i in range(len(imgs)):
        plt.imshow(imgs[i])
        plt.show()
        for category_mask in masks[i]:
            if np.max(np.array(category_mask)) != 0:
                plt.imshow(category_mask)
                plt.show()
    '''
if __name__ == '__main__':
    main()