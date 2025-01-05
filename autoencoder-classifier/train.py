import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Autoencoder, Classifier
from datasetloader import get_dataloaders
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_autoencoder(autoencoder, train_loader, device, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Autoencoder Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, _ in progress_bar:
            images = images.to(device)

            # Forward pass
            reconstructed = autoencoder(images)
            loss = criterion(reconstructed, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{running_loss / len(train_loader):.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
    return autoencoder

def train_classifier(autoencoder, classifier, train_loader, test_loader, device, num_epochs, learning_rate, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    best_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Classifier Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Get latent features from autoencoder
            with torch.no_grad():
                latent = autoencoder.encoder(images)

            # Forward pass through classifier
            outputs = classifier(latent)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{running_loss / len(train_loader):.4f}')

        # Evaluate on test set
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Classifier Evaluation Epoch {epoch+1}/{num_epochs}', leave=False):
                images, labels = images.to(device), labels.to(device)
                latent = autoencoder.encoder(images)
                outputs = classifier(latent)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%')

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': classifier.state_dict(),
                'best_acc': best_acc,
            }, 'best_classifier.pth')

def show_test_images(autoencoder, classifier, test_loader, device, class_names, num_images=20):
    autoencoder.eval()
    classifier.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        latent = autoencoder.encoder(images)
        outputs = classifier(latent)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        ax = plt.subplot(4, 5, i + 1)  # 4 rows x 5 columns
        ax.imshow(np.clip(images[i], 0, 1))  # Clipping to ensure valid pixel range
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root_dir = "/home/cha0s/Desktop/uas-tasks/Autoencoder-MC/freiburg_groceries_dataset/"
    classid_path = "/home/cha0s/Desktop/uas-tasks/Autoencoder-MC/freiburg_groceries_dataset/classid.txt"
    num_classes = 25

    # Hyperparameters
    latent_dim = 128
    batch_size = 32
    autoencoder_epochs = 50
    classifier_epochs = 50
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_dataloaders(root_dir, classid_path, batch_size)
    with open(classid_path, 'r') as f:
        class_names = [line.strip().split()[1] for line in f]

    # Initialize models
    autoencoder = Autoencoder(latent_dim).to(device)
    classifier = Classifier(latent_dim, num_classes).to(device)

    # Train autoencoder
    print("Training Autoencoder...")
    autoencoder = train_autoencoder(autoencoder, train_loader, device, autoencoder_epochs, learning_rate)

    # Train classifier
    print("Training Classifier...")
    train_classifier(autoencoder, classifier, train_loader, test_loader, device, classifier_epochs, learning_rate, class_names)
