import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import argparse

from dataloader import create_dataloaders
from model import ViT
from transform import draw_mouth

mp_face_mesh = mp.solutions.face_mesh


def scale_landmarks(landmarks):
    # Initialize a scaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    scaled_matrix = scaler.fit_transform(landmarks.reshape(-1, 1))

    # Reshape the scaled matrix back to its original shape
    scaled_matrix = scaled_matrix.reshape(landmarks.shape)

    return scaled_matrix

def evaluate(y_true, y_pred, scaler, lm_keys, dataset_name):
    save_path = f'samples/{dataset_name}/frames'
    os.makedirs(save_path, exist_ok=True)

    # reverse transform output for y_pred
    output_np = scaler.inverse_transform(y_pred.cpu().numpy())
    landmarks_np_pred = output_np.reshape(y_pred.shape[0], 20, 3)
    landmarks_np_pred = scale_landmarks(landmarks_np_pred)

    # reverse transform output for y_true
    output_np_true = scaler.inverse_transform(y_true.cpu().numpy())
    landmarks_np_true = output_np_true.reshape(y_true.shape[0], 20, 3)
    landmarks_np_true = scale_landmarks(landmarks_np_true)

    landmarks_pred = []
    landmarks_true = []
    for sample in range(landmarks_np_pred.shape[0]):
        sample_dict_pred = {}
        sample_dict_true = {}
        sample_reshaped_pred = landmarks_np_pred[sample].reshape(20, 3)
        sample_reshaped_true = landmarks_np_true[sample].reshape(20, 3)
        for idx, lm_key in enumerate(lm_keys):
            sample_dict_pred[lm_key] = sample_reshaped_pred[idx].tolist()
            sample_dict_true[lm_key] = sample_reshaped_true[idx].tolist()
        landmarks_pred.append(sample_dict_pred)
        landmarks_true.append(sample_dict_true)

    # draw mouth
    for idx, (landmark_pred, landmark_true) in enumerate(zip(landmarks_pred, landmarks_true)):

        image = np.ones((256, 256, 3), np.uint8) * 255
        image = draw_mouth(image, mp_face_mesh.FACEMESH_LIPS, landmark_pred, color=(0, 0, 255))
        image = draw_mouth(image, mp_face_mesh.FACEMESH_LIPS, landmark_true, color=(0, 255, 0)) # draw y_true in green color

        # save image
        cv2.imwrite(os.path.join(save_path, f'{idx}.png'), image)


def train(model, train_loader, val_loader, scaler, lm_keys, epochs, device, dataset_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).float()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    # Create a lists to store the losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, landmarks in train_loader:
            images, landmarks = images.to(device).float(), landmarks.to(device).float()

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, landmarks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, landmarks in val_loader:
                images, landmarks = images.to(device).float(), landmarks.to(device).float()
                y_pred = model(images)
                loss = criterion(y_pred, landmarks)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if epoch % 10 == 0:

            # create samples
            evaluate(landmarks, y_pred, scaler, lm_keys, dataset_name=dataset_name)

            # Create the plots
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.title('Losses over time')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')

            # Ensure the output directory exists
            os.makedirs(f'samples/{dataset_name}', exist_ok=True)

            # Save the plot
            plt.savefig(f'samples/{dataset_name}/losses.png')

            # save model
            model_path = f'models/{dataset_name}/'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))


def main(dataset_name, batch_size, epochs):
    train_loader, val_loader, scaler, lm_keys = create_dataloaders(dataset_name, batch_size)
    model = ViT()
    train(model, train_loader, val_loader, scaler, lm_keys, epochs=epochs)# python dataset.py 'example.mov' 'dataset_name', dataset_name=dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Vision Transformer (ViT) model on a dataset.")
    parser.add_argument('dataset_name', type=str, help='The name of the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training. Default is 32.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training. Default is 30.')

    args = parser.parse_args()

    main(args.dataset_name, args.batch_size, args.epochs)