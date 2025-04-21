"""
This module trains a deep learning model for gesture classification using facial 
landmarks and images. It includes data loading, model training, validation, 
and early stopping.
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data_loader import GestureDataset
from model import GestureClassifier
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


num_classes = 5
labels = ['left_tilt', 'right_tilt', 'frown', 'mouth_open', "front_facing"]

DATA_PATH = r'dataset\images_landmarks_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transforms ---
  transforms.ToTensor(),
  transforms.Resize((224, 224)),
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


dataset = GestureDataset(
    num_classes = num_classes,
    labels = labels,
    dataset_folder_path = "dataset\images_landmarks_dataset", 
    transform=train_transforms)
# Total dataset size
dataset_size = len(dataset)

# Calculate 80/20 split
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

# Split the dataset
train_set, test_set = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_set, batch_size=50,shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1,shuffle=True)

model = GestureClassifier(landmark_dim=478, num_classes=num_classes).to(device)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_dataloader)):
        input_tensor, labels, landmark = data
        images = input_tensor.to(device).float()
        labels = labels.to(device)
        landmarks = landmark.to(device).float()
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(images, landmarks)

        # Compute the loss and its gradientsQ
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

def main(model_save_path:str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/facial_gesture_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100
    best_vloss = float('inf')
    
    # --- Early stopping parameters ---
    patience = 5
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in tqdm(enumerate(test_dataloader)):
                vinputs, vlabels, vlandmark = vdata
                vinputs = vinputs.to(device).float()
                vlabels = vlabels.to(device)
                vlandmark = vlandmark.to(device).float()
                voutputs = model(vinputs, vlandmark)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            early_stop_counter = 0
            model_path = os.path.join(model_save_path, 'model_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1
            print(
                    f'Validation loss did not improve. '
                    f'Early stop counter: {early_stop_counter}/{patience}'
                )
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        epoch_number += 1


if __name__ == '__main__':
    main(model_save_path = r"models\updated_weights")
    print("Training complete.")