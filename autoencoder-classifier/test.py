import torch
from model import AutoencoderClassifier
from datasetloader import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def show_test_images(images, labels, predictions, class_mapping):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    labels, predictions = labels.cpu().numpy(), predictions.cpu().numpy()
    
    num_images = min(50, len(images))
    rows, cols = 5, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle("Test Images: True vs Predicted Labels", fontsize=16, y=0.92)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(np.clip(images[i], 0, 1))
            true_label = class_mapping.get(labels[i], labels[i])
            pred_label = class_mapping.get(predictions[i], predictions[i])
            ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device):
    correct, total = 0, 0
    all_images, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if len(all_images) < 50:
                all_images.append(images.cpu())
                all_labels.append(labels.cpu())
                all_preds.append(predicted.cpu())

    accuracy = 100 * correct / total

    if all_images:
        display_images = torch.cat(all_images)[:50]
        display_labels = torch.cat(all_labels)[:50]
        display_preds = torch.cat(all_preds)[:50]
        return accuracy, (display_images, display_labels, display_preds)

    return accuracy, (None, None, None)

def test_model(root_dir, classid_path, num_classes, num_iterations=10, latent_dim=128, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(classid_path, 'r') as f:
        class_mapping = {int(cid): name for name, cid in (line.split() for line in f)}

    accuracies = []

    for _ in tqdm(range(num_iterations), desc="Testing iterations"):
        _, test_loader = get_dataloaders(root_dir, classid_path, batch_size)
        model = AutoencoderClassifier(latent_dim, num_classes).to(device)

        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        accuracy, display_data = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)

        if len(accuracies) == 1 and display_data[0] is not None:
            show_test_images(*display_data, class_mapping)

    mean_acc = sum(accuracies) / len(accuracies)

    return mean_acc

if __name__ == "__main__":
    mean_acc = test_model(
        root_dir="/home/cha0s/Desktop/uas-tasks/Autoencoder-MC/freiburg_groceries_dataset/",
        classid_path="/home/cha0s/Desktop/uas-tasks/Autoencoder-MC/freiburg_groceries_dataset/classid.txt",
        num_classes=25
    )
    print(f"Mean Accuracy: {mean_acc:.2f}%")
