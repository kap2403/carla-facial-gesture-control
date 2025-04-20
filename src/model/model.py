"""
This module has the gesture classification model using pretrained ResNet50 and 
MLP for landmark features.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class GestureClassifier(nn.Module):
    def __init__(self, landmark_dim=478, num_classes=5):
        super(GestureClassifier, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        # Remove the last FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  

        # MLP
        self.landmark_fc = nn.Sequential(
            nn.Linear(1434, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )  # Output: (B, 64)

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, landmarks):
        # Image: (B, 3, 224, 224)
        x_img = self.resnet(image)              
        x_img = x_img.view(x_img.size(0), -1)   

        # Landmarks: (B, 136)
        x_lm = self.landmark_fc(landmarks)     

        # Concatenate
        x = torch.cat([x_img, x_lm], dim=1)     
        out = self.classifier(x)               

        return out
