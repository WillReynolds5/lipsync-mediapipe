import os
import pickle

import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


def extract_values(nested_dict):

    samples = []
    for image_key in nested_dict:
        feature_lst = []
        for landmark_key in nested_dict[image_key]:
            feature_lst.append(nested_dict[image_key][landmark_key])
        features_np = np.array(feature_lst)
        features_flat_np = features_np.reshape(-1)
        samples.append(features_flat_np.tolist())
    return np.array(samples)


class LandmarksMelSpectrogramDataset(Dataset):

    def __init__(self, dataset_name, keys, landmarks, transform=None):
        self.dataset_name = dataset_name
        self.transform = transform

        self.landmarks = landmarks
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        landmark = self.landmarks[index]

        # load corresponding mel-spectrogram image
        image_path = os.path.join('data', self.dataset_name, 'lipsync', 'melspectrogram', f'{key}.png')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, landmark


def get_landmark_keys(landmarks):
    return list(landmarks[list(landmarks.keys())[0]].keys())

def create_dataloaders(dataset_name, batch_size):
    transform = transforms.Compose([
        # transforms.Resize((32, 128)),
        transforms.Lambda(lambda image: image.convert("RGB")),  # Ensure the image is 3-channel RGB
        transforms.ToTensor()
    ])

    # Load and scale all the data together
    with open(f'data/{dataset_name}/lipsync/landmarks.pkl', 'rb') as f:
        landmarks = pickle.load(f)

    keys = list(landmarks.keys())
    length = len(keys)
    lm_keys = get_landmark_keys(landmarks)

    # Fit scaler to all the data
    scaler = StandardScaler()
    all_data = extract_values(landmarks)
    scaler.fit(all_data)
    transformed_data = scaler.transform(all_data)
    with open(f'data/{dataset_name}/lipsync/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split into training and validation sets
    train_keys = keys[:int(0.8 * length)]
    val_keys = keys[int(0.8 * length):]

    train_landmarks = transformed_data[:int(0.8 * length)]
    val_landmarks = transformed_data[int(0.8 * length):]

    train_dataset = LandmarksMelSpectrogramDataset(dataset_name, train_keys, train_landmarks, transform=transform)
    val_dataset = LandmarksMelSpectrogramDataset(dataset_name, val_keys, val_landmarks, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, scaler, lm_keys

