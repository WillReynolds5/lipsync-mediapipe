import os
import pickle

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def align_and_center_3d_coords(data):
    point1 = np.array(data[78])
    point2 = np.array(data[308])

    # Calculate the vector from point1 to point2
    vec = point2 - point1

    # Normalize the vector
    vec = vec / np.linalg.norm(vec)

    # Define the target vector - we want to align vec with the x-axis [1, 0, 0]
    target = np.array([1, 0, 0])

    # Compute the rotation axis using cross product
    axis = np.cross(vec, target)
    axis_norm = np.linalg.norm(axis)

    # If the points are already aligned, no need to do anything further
    if axis_norm == 0:
        return data

    # Normalize the rotation axis
    axis /= axis_norm

    # Compute the rotation angle using dot product
    angle = np.arccos(np.dot(vec, target))

    # Compute the rotation matrix using Rodriguez's formula
    u = axis
    R = np.cos(angle) * np.eye(3) + np.sin(angle) * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) + (
                1 - np.cos(angle)) * np.outer(u, u)

    # Apply the rotation to all points and compute the center of gravity
    center = np.zeros(3)
    for key in data.keys():
        rotated_point = np.dot(R, np.array(data[key]) - point1) + point1
        data[key] = list(rotated_point)
        center += rotated_point

    # Center of gravity
    center /= len(data)

    # Centering the data: subtract the center of gravity from all points
    for key in data.keys():
        data[key] = list(np.array(data[key]) - center)

    return data


def draw_mouth(image, connections, landmark, color=(0, 0, 255), thickness=1):

    width = image.shape[1]
    height = image.shape[0]

    for connection in connections:
        try:
            start_point = landmark[connection[0]]
            end_point = landmark[connection[1]]
            start_point = (int(start_point[0]*width), int(start_point[1]*height))
            end_point = (int(end_point[0]*width), int(end_point[1]*height))
            cv2.line(image, start_point, end_point, color, thickness)
        except:
            pass

    return image


if __name__ == '__main__':

    # Load data (deserialize)
    with open('data/tucker/lipsync/landmarks.pkl', 'rb') as handle:
        facemesh_data = pickle.load(handle)

    random_key = np.random.choice(list(facemesh_data.keys()), 20)
    save_path = 'testing'
    os.makedirs(save_path, exist_ok=True)

    for key in random_key:
        landmark = align_and_center_3d_coords(facemesh_data[key])

        blank = np.ones((256, 256, 3), np.uint8) * 255
        mouth_img = draw_mouth(blank, mp_face_mesh.FACEMESH_LIPS, landmark)

        # save image
        cv2.imwrite(os.path.join(save_path, f'{key}.png'), mouth_img)
