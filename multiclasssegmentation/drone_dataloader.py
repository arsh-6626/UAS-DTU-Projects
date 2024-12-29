import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from UNET import *

class DroneDataset(Dataset):
    def __init__(self, img_dir, label_dir, label2class, transform=None, augmentation=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label2class = label2class
        self.transform = transform
        self.augmentation = augmentation

        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        
        img = Image.open(img_path).convert("RGB")  
        label = Image.open(label_path).convert("RGB") 

        
        img = img.resize((512, 512))
        label = label.resize((512, 512))

    
        mapped_label = np.zeros((label.size[1], label.size[0]), dtype=np.uint8)
        for color, class_index in self.label2class.items():
            mask = np.all(np.array(label) == color, axis=-1)
            mapped_label[mask] = class_index

        label = torch.from_numpy(mapped_label).long()  

        if self.augmentation:
            img, label = self.apply_augment(img, label)


        if self.transform:
            img = self.transform(img)

        label = label.long()
        return img, label

    def apply_augment(self, image, label):
        augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
        ])
        image = augmentation(image)
        label = augmentation(label)
        return image, label

if __name__ == "__main__":
    color_to_class = {
        (155, 38, 182): 0,  
        (14, 135, 204): 1,  
        (124, 252, 0): 2,   
        (255, 20, 147): 3,  
        (169, 169, 169): 4  
    }

    image_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DroneDataset(
        img_dir="/home/cha0s/Desktop/uas-tasks/PyTorch-MCS/original_images"
        label_dir="/home/cha0s/Desktop/uas-tasks/PyTorch-MCS/label_images_semantic",  
        label2class=color_to_class,
        transform=image_transform,
        augmentation=True
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for img, label in dataloader:
        print(img.shape, label.shape)
