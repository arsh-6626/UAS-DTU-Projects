import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FreiburgDataset(Dataset):
    
    def __init__(self, root_dir, classid_path, transform = None, train = True, augmentation=False):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
        self.augmentation = augmentation

        self.class2id = {}

        with open(classid_path, 'r') as f:
            for line in f:
                class_name, class_id = line.strip().split()
                self.class2id[class_name] = int(class_id)

        self.images = []
        self.labels = []

        for class_name in os.listdir(os.path.join(root_dir, "images")):
            class_path = os.path.join(root_dir, 'images', class_name)
            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(self.class2id[class_name])

        n_samples = len(self.images)
        indices = torch.randperm(n_samples)
        split = int(0.8*n_samples)

        if train:
            self.images = [self.images[i] for i in indices[:split]]
            self.labels = [self.labels[i] for i in indices[:split]]
        else:
            self.images = [self.images[i] for i in indices[split:]]
            self.labels = [self.labels[i] for i in indices[split:]]

    def __len__(self):
        return len(self.images)

    def apply_augment(self, image):
        augmentation = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(degrees=30),
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1))
        transforms.RandomPerspective(
                distortion_scale=0.15, 
                p=0.3                   
            ),
        transforms.RandomAffine(
                degrees=0,        
                translate=(0.1, 0.1),  
                scale=(0.95, 1.05),    
                fill=1                  
            )
        ])
        image = augmentation(image)
        return image

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.augmentation:
            image= self.apply_augment(image)

        if self.transform:
            image = self.transform(image)

        return image, label
            

        

def get_dataloaders(root_dir, classid_path, batch_size=32, num_workers=4):
    train_dataset = FreiburgDataset(root_dir, classid_path, train=True)
    test_dataset = FreiburgDataset(root_dir, classid_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers)

    return train_loader, test_loader