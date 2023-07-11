import os
import pickle

import argparse
import cv2
import ffmpeg
import librosa
import librosa.display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from transform import align_and_center_3d_coords
from utils.mesh_key import lips

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


def extract_audio_from_video(video_path):
    # Derive audio_path from video_path
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}.wav"

    # Extract audio from video using ffmpeg
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    return audio_path

def generate_mel_spectrogram(audio_path):
    # Load audio file with librosa
    y, sr = librosa.load(audio_path)
    # Generate Mel-scaled power spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # Convert to log scale (dB). Use the peak power as reference.
    mel_spectrogram = librosa.power_to_db(S, ref=np.max)
    left_pad, right_pad = np.full((128, 16), -80), np.full((128, 16), -80)
    mel_spectrogram = np.concatenate((left_pad, mel_spectrogram, right_pad), axis=1)
    # Normalize each frequency band independently
    min_vals = np.min(mel_spectrogram, axis=1)[:, np.newaxis]
    max_vals = np.max(mel_spectrogram, axis=1)[:, np.newaxis]
    mel_spectrogram = (mel_spectrogram - min_vals) / (max_vals - min_vals)

    # Apply colormap
    mel_spectrogram = cm.viridis(mel_spectrogram)[:, :, :3]

    return mel_spectrogram[:, :, :3]


def extract_landmarks(frame):

    # Convert the BGR image to RGB before processing
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = {}
    if results.multi_face_landmarks:
        for face_landmarks_mp in results.multi_face_landmarks:
            face_landmarks = face_landmarks_mp.landmark
            for idx in lips:
                vertex = face_landmarks[idx]
                xyz = [vertex.x, vertex.y, vertex.z]
                landmarks[idx] = xyz
    return landmarks


def process_video(video_path, dataset_name):

    # Extract audio from video
    audio_path = extract_audio_from_video(video_path)

    # Generate mel spectrogram from audio
    mel_spectrogram = generate_mel_spectrogram(audio_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize dataset
    slice_width = 32
    slc_multiplier = slice_width // 2
    current_frame = 0

    # Initialize landmarks dictionary
    landmarks_dict = {}

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            if current_frame > slc_multiplier and current_frame < total_frames - slc_multiplier:

                # Extract landmarks
                landmarks = extract_landmarks(frame)
                aligned_landmarks = align_and_center_3d_coords(landmarks)
                # Add landmarks to dictionary
                landmarks_dict[current_frame] = aligned_landmarks

                # Calculate corresponding spectrogram slice
                frame_width_in_spectrogram = mel_spectrogram.shape[1] / total_frames
                start = int(current_frame * frame_width_in_spectrogram) - slc_multiplier
                end = int(current_frame * frame_width_in_spectrogram) + slc_multiplier
                spectrogram_slice = mel_spectrogram[:, start:end]

                # Create directory path if it doesn't exist
                directory_path = os.path.join('data', dataset_name, 'melspectrogram')
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

                # Normalize the spectrogram slice to 0-1
                norm_slice = (spectrogram_slice - np.min(spectrogram_slice)) / (np.max(spectrogram_slice) - np.min(spectrogram_slice))

                # Apply colormap (like viridis)
                image_slice = cm.viridis(norm_slice)

                # Save the spectrogram slice as an image
                plt.imsave(os.path.join(directory_path, f'{current_frame}.png'), image_slice)

            # Increment the frame count
            current_frame += 1
        else:
            break

    cap.release()

    # Save landmarks dictionary as a pickle file
    pickle_directory_path = os.path.join('data', dataset_name)
    with open(os.path.join(pickle_directory_path, 'landmarks.pkl'), 'wb') as handle:
        pickle.dump(landmarks_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Return the dataset
    return landmarks_dict


def main(video_path, dataset_name):
    # Run the function
    dataset = process_video(video_path, dataset_name)
    print(f"Dataset: {dataset_name}, Length: {len(dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video and audio into a dataset.")
    parser.add_argument('video_path', type=str, help='The path to the video file.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset.')
    args = parser.parse_args()

    main(args.video_path, args.dataset_name)

# python dataset.py 'example.mov' 'dataset_name'