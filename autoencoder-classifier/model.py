import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_channels=256):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_maps = self.encoder(x)
        reconstruction = self.decoder(latent_maps)
        return reconstruction

class Classifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(32, num_classes))
    
    def forward(self, x):
        x = self.classifier(x)
        return x

class AutoencoderClassifier(nn.Module):
    def __init__(self, latent_channels=256, num_classes=25):
        super(AutoencoderClassifier, self).__init__()
        self.autoencoder = Autoencoder(latent_channels=latent_channels)
        self.classifier = Classifier(input_channels=latent_channels, num_classes=num_classes)

    def forward(self, x):
        latent_maps = self.autoencoder.encoder(x)
        reconstruction = self.autoencoder.decoder(latent_maps)
        classification = self.classifier(latent_maps)
        return reconstruction, classification

    def encode(self, x):
        return self.autoencoder.encoder(x)