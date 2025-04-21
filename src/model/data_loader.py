import torch
from torch.utils.data import Dataset
import os
import csv
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import os
import csv
import cv2
import numpy as np

class GestureDataset(Dataset):
    def __init__(self, num_classes, labels,  dataset_folder_path, transform=None):
        self.labels = []
        self.images = []
        self.face_landmarks = []
        self.transform = transform  # Store transform
        self.num_classes = num_classes  # Number of classes
        # Create a dictionary mapping
        label_to_int = {label: idx for idx, label in enumerate(labels)}

        for folder_name in os.listdir(dataset_folder_path):
            sub_folder_path = os.path.join(dataset_folder_path, folder_name)
            
            if os.path.isdir(sub_folder_path):
                csv_file = os.path.join(sub_folder_path, f"{folder_name}_coords.csv")
                image_folder = os.path.join(sub_folder_path, "images")

                if not os.path.exists(csv_file):
                    print(f"Warning: CSV file not found in {sub_folder_path}")
                    continue

                with open(csv_file, 'r') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        label = label_to_int[row[0]]  # Convert label to int
                        image_name = row[1]
                        landmarks = list(map(float, row[2:]))

                        image_path = os.path.join(image_folder, image_name)

                        if os.path.exists(image_path):
                            self.labels.append(label)
                            self.images.append(image_path)
                            self.face_landmarks.append(landmarks)
                        else:
                            print(f"Image not found: {image_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.labels[idx]
        landmarks = self.face_landmarks[idx]

        if self.transform is not None:
            image = self.transform(image)
        # One-hot encoding for 5 classes
        target_one_hot = np.eye(self.num_classes)[target]
        label = torch.tensor(target_one_hot, dtype=torch.float32)
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

        return image, label, landmarks_tensor