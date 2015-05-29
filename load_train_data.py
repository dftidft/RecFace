import cv2
import numpy as np

def load_train_data():

    landmark_path = 'D:/Dataset/face/Face_landmark_muct/muct-landmarks/muct76-opencv.csv'
    landmark_file = open(landmark_path)
    lines = landmark_file.readlines()
    train_data = []
    for line in lines[1:]:
        landmark = []
        landmark_arr = line.split(',')[2:]
        for x in landmark_arr:
            landmark.append(float(x))
        train_data.append(landmark)

    return train_data


if __name__ == '__main__':
    train_data = load_train_data()



