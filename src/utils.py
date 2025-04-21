import torch
import pickle
from typing import Tuple, List
import numpy as np
import cv2
from model.model import GestureClassifier
from torchvision import transforms

def load_model(model_path: str):
    """
    Load a pre-trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        _type_: loaded model.
    """

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_dnn_model(PATH: str, num_classes: int) -> GestureClassifier:
    """
    Load a pre-trained DNN model from a file.
    This function loads a model architecture and its weights from the specified path.

    Args:
        PATH (str): Path to the model file.
        num_classes (int): Number of classes for the model.

    Returns:
        GestureClassifier: loaded model.
    """

    saved_model = GestureClassifier(num_classes=num_classes)
    saved_model.load_state_dict(torch.load(PATH))
    return saved_model

def data_preprocessing(image: np.array, 
                       landmarks_array: np.array
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses the image and landmarks for model input.

    Args:
        image (np.array): image to be preprocessed 
        landmarks_array (np.array): facial landmarks to be preprocessed to tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: image and landmarks tensors ready for model input.
    """
    train_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = train_transforms(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    landmarks_array = torch.tensor(landmarks_array, dtype=torch.float32).unsqueeze(0)
    return img_tensor, landmarks_array