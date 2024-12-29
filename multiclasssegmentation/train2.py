import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter
from tqdm import tqdm
import matplotlib.pyplot as plt

class DroneSegmentation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config['model'].to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        self.scaler = torch.cuda.amp.GradScaler()
        
    def setup_data(self, dataset):
        """Setup train and test dataloaders"""
        train_size = int(self.config['train_split'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Disable augmentation for test dataset
        test_dataset.dataset.augmentation = False
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        loop = tqdm(self.train_loader)
        running_loss = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                loss = self.loss_fn(predictions, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (batch_idx + 1))
        
        return running_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(data)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                _, predicted = torch.max(predictions, 1)

                correct += (predicted == targets).sum().item()
                total += targets.numel()

        return correct / total
    
    def visualize_predictions(self, num_images=3):
        """Visualize model predictions"""
        self.model.eval()
        images, targets = next(iter(self.test_loader))
        images = images.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            predictions = self.model(images)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            _, predicted = torch.max(predictions, 1)

        for i in range(min(num_images, len(images))):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            img = images[i].cpu().permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            axs[1].imshow(targets[i].cpu(), cmap='tab10')
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')
            axs[2].imshow(predicted[i].cpu(), cmap='tab10')
            axs[2].set_title("Prediction")
            axs[2].axis('off')

            plt.show()
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config['num_epochs']):
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            avg_loss = self.train_epoch()
            print(f"Average Loss: {avg_loss:.4f}")
            accuracy = self.evaluate()
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            self.visualize_predictions(num_images=3)

if __name__ == "__main__":
    from UNET import UNET
    from drone_dataloader import DroneDataset
    
    config = {
        'model': UNET(in_channels=3, out_channels=5),
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 10,
        'num_workers': 2,
        'pin_memory': True,
        'train_split': 0.90,
        'image_height': 512,
        'image_width': 512
    }
    
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
        img_dir="/home/cha0s/Desktop/uas-tasks/PyTorch-MCS/original_images",
        label_dir="/home/cha0s/Desktop/uas-tasks/PyTorch-MCS/label_images_semantic",
        label2class=color_to_class,
        transform=image_transform,
        augmentation=True
    )
    
    # Initialize trainer and start training
    trainer = DroneSegmentation(config)
    trainer.setup_data(dataset)
    trainer.train()