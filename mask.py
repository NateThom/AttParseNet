import os
import cv2
import tqdm

import numpy as np
import pandas as pd


def chin(image_name, landmarks_df):
    # Chin
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{5}'], landmarks_df.loc[image_name, f'y_{5}']],
            [landmarks_df.loc[image_name, f'x_{6}'], landmarks_df.loc[image_name, f'y_{6}']],
            [landmarks_df.loc[image_name, f'x_{7}'], landmarks_df.loc[image_name, f'y_{7}']],
            [landmarks_df.loc[image_name, f'x_{8}'], landmarks_df.loc[image_name, f'y_{8}']],
            [landmarks_df.loc[image_name, f'x_{9}'], landmarks_df.loc[image_name, f'y_{9}']],
            [landmarks_df.loc[image_name, f'x_{10}'], landmarks_df.loc[image_name, f'y_{10}']],
            [landmarks_df.loc[image_name, f'x_{11}'], landmarks_df.loc[image_name, f'y_{11}']],
            [landmarks_df.loc[image_name, f'x_{54}'], landmarks_df.loc[image_name, f'y_{54}']],
            [landmarks_df.loc[image_name, f'x_{55}'], landmarks_df.loc[image_name, f'y_{55}']],
            [landmarks_df.loc[image_name, f'x_{56}'], landmarks_df.loc[image_name, f'y_{56}']],
            [landmarks_df.loc[image_name, f'x_{57}'], landmarks_df.loc[image_name, f'y_{57}']],
            [landmarks_df.loc[image_name, f'x_{58}'], landmarks_df.loc[image_name, f'y_{58}']],
            [landmarks_df.loc[image_name, f'x_{59}'], landmarks_df.loc[image_name, f'y_{59}']],
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']]
        ]
    ], dtype=np.int32)
    return points


def left_cheek(image_name, landmarks_df):
    # Left Cheek
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{0}'], landmarks_df.loc[image_name, f'y_{0}']],
            [landmarks_df.loc[image_name, f'x_{1}'], landmarks_df.loc[image_name, f'y_{1}']],
            [landmarks_df.loc[image_name, f'x_{2}'], landmarks_df.loc[image_name, f'y_{2}']],
            [landmarks_df.loc[image_name, f'x_{3}'], landmarks_df.loc[image_name, f'y_{3}']],
            [landmarks_df.loc[image_name, f'x_{4}'], landmarks_df.loc[image_name, f'y_{4}']],
            [landmarks_df.loc[image_name, f'x_{5}'], landmarks_df.loc[image_name, f'y_{5}']],
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']],
            [landmarks_df.loc[image_name, f'x_{31}'], landmarks_df.loc[image_name, f'y_{31}']],
            [landmarks_df.loc[image_name, f'x_{39}'], landmarks_df.loc[image_name, f'y_{39}']],
            [landmarks_df.loc[image_name, f'x_{40}'], landmarks_df.loc[image_name, f'y_{40}']],
            [landmarks_df.loc[image_name, f'x_{41}'], landmarks_df.loc[image_name, f'y_{41}']],
            [landmarks_df.loc[image_name, f'x_{36}'], landmarks_df.loc[image_name, f'y_{36}']],
            [landmarks_df.loc[image_name, f'x_{0}'], landmarks_df.loc[image_name, f'y_{0}']],
        ]
    ], dtype=np.int32)
    return points


def right_cheek(image_name, landmarks_df):
    # Right Cheek
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{16}'], landmarks_df.loc[image_name, f'y_{16}']],
            [landmarks_df.loc[image_name, f'x_{15}'], landmarks_df.loc[image_name, f'y_{15}']],
            [landmarks_df.loc[image_name, f'x_{14}'], landmarks_df.loc[image_name, f'y_{14}']],
            [landmarks_df.loc[image_name, f'x_{13}'], landmarks_df.loc[image_name, f'y_{13}']],
            [landmarks_df.loc[image_name, f'x_{12}'], landmarks_df.loc[image_name, f'y_{12}']],
            [landmarks_df.loc[image_name, f'x_{11}'], landmarks_df.loc[image_name, f'y_{11}']],
            [landmarks_df.loc[image_name, f'x_{54}'], landmarks_df.loc[image_name, f'y_{54}']],
            [landmarks_df.loc[image_name, f'x_{35}'], landmarks_df.loc[image_name, f'y_{35}']],
            [landmarks_df.loc[image_name, f'x_{42}'], landmarks_df.loc[image_name, f'y_{42}']],
            [landmarks_df.loc[image_name, f'x_{47}'], landmarks_df.loc[image_name, f'y_{47}']],
            [landmarks_df.loc[image_name, f'x_{46}'], landmarks_df.loc[image_name, f'y_{46}']],
            [landmarks_df.loc[image_name, f'x_{45}'], landmarks_df.loc[image_name, f'y_{45}']],
            [landmarks_df.loc[image_name, f'x_{16}'], landmarks_df.loc[image_name, f'y_{16}']],
        ]
    ], dtype=np.int32)
    return points


def upper_lip(image_name, landmarks_df):
    # Upper Lip
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']],
            [landmarks_df.loc[image_name, f'x_{31}'], landmarks_df.loc[image_name, f'y_{31}']],
            [landmarks_df.loc[image_name, f'x_{32}'], landmarks_df.loc[image_name, f'y_{32}']],
            [landmarks_df.loc[image_name, f'x_{33}'], landmarks_df.loc[image_name, f'y_{33}']],
            [landmarks_df.loc[image_name, f'x_{34}'], landmarks_df.loc[image_name, f'y_{34}']],
            [landmarks_df.loc[image_name, f'x_{35}'], landmarks_df.loc[image_name, f'y_{35}']],
            [landmarks_df.loc[image_name, f'x_{54}'], landmarks_df.loc[image_name, f'y_{54}']],
            [landmarks_df.loc[image_name, f'x_{53}'], landmarks_df.loc[image_name, f'y_{53}']],
            [landmarks_df.loc[image_name, f'x_{52}'], landmarks_df.loc[image_name, f'y_{52}']],
            [landmarks_df.loc[image_name, f'x_{51}'], landmarks_df.loc[image_name, f'y_{51}']],
            [landmarks_df.loc[image_name, f'x_{50}'], landmarks_df.loc[image_name, f'y_{50}']],
            [landmarks_df.loc[image_name, f'x_{49}'], landmarks_df.loc[image_name, f'y_{49}']],
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']]
        ]
    ], dtype=np.int32)
    return points


def mouth(image_name, landmarks_df):
    # Mouth
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']],
            [landmarks_df.loc[image_name, f'x_{49}'], landmarks_df.loc[image_name, f'y_{49}']],
            [landmarks_df.loc[image_name, f'x_{50}'], landmarks_df.loc[image_name, f'y_{50}']],
            [landmarks_df.loc[image_name, f'x_{51}'], landmarks_df.loc[image_name, f'y_{51}']],
            [landmarks_df.loc[image_name, f'x_{52}'], landmarks_df.loc[image_name, f'y_{52}']],
            [landmarks_df.loc[image_name, f'x_{53}'], landmarks_df.loc[image_name, f'y_{53}']],
            [landmarks_df.loc[image_name, f'x_{54}'], landmarks_df.loc[image_name, f'y_{54}']],
            [landmarks_df.loc[image_name, f'x_{55}'], landmarks_df.loc[image_name, f'y_{55}']],
            [landmarks_df.loc[image_name, f'x_{56}'], landmarks_df.loc[image_name, f'y_{56}']],
            [landmarks_df.loc[image_name, f'x_{57}'], landmarks_df.loc[image_name, f'y_{57}']],
            [landmarks_df.loc[image_name, f'x_{58}'], landmarks_df.loc[image_name, f'y_{58}']],
            [landmarks_df.loc[image_name, f'x_{59}'], landmarks_df.loc[image_name, f'y_{59}']],
            [landmarks_df.loc[image_name, f'x_{48}'], landmarks_df.loc[image_name, f'y_{48}']]
        ]
    ], dtype=np.int32)
    return points


def left_eye(image_name, landmarks_df):
    # L Eye
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{36}'], landmarks_df.loc[image_name, f'y_{36}']],
            [landmarks_df.loc[image_name, f'x_{37}'], landmarks_df.loc[image_name, f'y_{37}']],
            [landmarks_df.loc[image_name, f'x_{38}'], landmarks_df.loc[image_name, f'y_{38}']],
            [landmarks_df.loc[image_name, f'x_{39}'], landmarks_df.loc[image_name, f'y_{39}']],
            [landmarks_df.loc[image_name, f'x_{40}'], landmarks_df.loc[image_name, f'y_{40}']],
            [landmarks_df.loc[image_name, f'x_{41}'], landmarks_df.loc[image_name, f'y_{41}']],
            [landmarks_df.loc[image_name, f'x_{36}'], landmarks_df.loc[image_name, f'y_{36}']]
        ]
    ], dtype=np.int32)
    return points


def right_eye(image_name, landmarks_df):
    # R Eye
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{42}'], landmarks_df.loc[image_name, f'y_{42}']],
            [landmarks_df.loc[image_name, f'x_{43}'], landmarks_df.loc[image_name, f'y_{43}']],
            [landmarks_df.loc[image_name, f'x_{44}'], landmarks_df.loc[image_name, f'y_{44}']],
            [landmarks_df.loc[image_name, f'x_{45}'], landmarks_df.loc[image_name, f'y_{45}']],
            [landmarks_df.loc[image_name, f'x_{46}'], landmarks_df.loc[image_name, f'y_{46}']],
            [landmarks_df.loc[image_name, f'x_{47}'], landmarks_df.loc[image_name, f'y_{47}']],
            [landmarks_df.loc[image_name, f'x_{42}'], landmarks_df.loc[image_name, f'y_{42}']]
        ]
    ], dtype=np.int32)
    return points


def nose(image_name, landmarks_df):
    # Nose
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{31}'] - (
                    landmarks_df.loc[image_name, f'x_{32}'] - landmarks_df.loc[image_name, f'x_{31}']),
             landmarks_df.loc[image_name, f'y_{31}']],
            [landmarks_df.loc[image_name, f'x_{32}'], landmarks_df.loc[image_name, f'y_{32}']],
            [landmarks_df.loc[image_name, f'x_{33}'], landmarks_df.loc[image_name, f'y_{33}']],
            [landmarks_df.loc[image_name, f'x_{34}'], landmarks_df.loc[image_name, f'y_{34}']],
            [landmarks_df.loc[image_name, f'x_{35}'] + (
                    landmarks_df.loc[image_name, f'x_{35}'] - landmarks_df.loc[image_name, f'x_{34}']),
             landmarks_df.loc[image_name, f'y_{35}']],
            [landmarks_df.loc[image_name, f'x_{42}'], landmarks_df.loc[image_name, f'y_{42}']],
            [landmarks_df.loc[image_name, f'x_{27}'], landmarks_df.loc[image_name, f'y_{27}']],
            [landmarks_df.loc[image_name, f'x_{39}'], landmarks_df.loc[image_name, f'y_{39}']],
            [landmarks_df.loc[image_name, f'x_{31}'] - (
                    landmarks_df.loc[image_name, f'x_{32}'] - landmarks_df.loc[image_name, f'x_{31}']),
             landmarks_df.loc[image_name, f'y_{31}']],
        ]
    ], dtype=np.int32)
    return points


def left_eye_brow(image_name, landmarks_df):
    # Left Eye Brow
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{17}'], landmarks_df.loc[image_name, f'y_{17}']],
            [landmarks_df.loc[image_name, f'x_{18}'], landmarks_df.loc[image_name, f'y_{18}']],
            [landmarks_df.loc[image_name, f'x_{19}'], landmarks_df.loc[image_name, f'y_{19}']],
            [landmarks_df.loc[image_name, f'x_{20}'], landmarks_df.loc[image_name, f'y_{20}']],
            [landmarks_df.loc[image_name, f'x_{21}'], landmarks_df.loc[image_name, f'y_{21}']],
        ]
    ], dtype=np.int32)
    return points


def right_eye_brow(image_name, landmarks_df):
    # Right Eye Brow
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{22}'], landmarks_df.loc[image_name, f'y_{22}']],
            [landmarks_df.loc[image_name, f'x_{23}'], landmarks_df.loc[image_name, f'y_{23}']],
            [landmarks_df.loc[image_name, f'x_{24}'], landmarks_df.loc[image_name, f'y_{24}']],
            [landmarks_df.loc[image_name, f'x_{25}'], landmarks_df.loc[image_name, f'y_{25}']],
            [landmarks_df.loc[image_name, f'x_{26}'], landmarks_df.loc[image_name, f'y_{26}']],
        ]
    ], dtype=np.int32)
    return points


def top_of_head(image_name, landmarks_df):
    # Top of Head
    if (landmarks_df.loc[image_name, f'y_{24}'] - (
            int(landmarks_df.loc[image_name, f'y_{8}'] * 1.1) - landmarks_df.loc[image_name, f'y_{33}']) > 0) and (
            landmarks_df.loc[image_name, f'y_{19}'] - (
            int(landmarks_df.loc[image_name, f'y_{8}'] * 1.1) - landmarks_df.loc[image_name, f'y_{33}']) > 0):
        points = np.array([
            [
                [landmarks_df.loc[image_name, f'x_{24}'], landmarks_df.loc[image_name, f'y_{24}']],
                [landmarks_df.loc[image_name, f'x_{24}'], landmarks_df.loc[image_name, f'y_{24}'] - (
                        int(landmarks_df.loc[image_name, f'y_{8}'] * 1.1) - landmarks_df.loc[image_name, f'y_{33}'])],
                [landmarks_df.loc[image_name, f'x_{19}'], landmarks_df.loc[image_name, f'y_{19}'] - (
                        int(landmarks_df.loc[image_name, f'y_{8}'] * 1.1) - landmarks_df.loc[image_name, f'y_{33}'])],
                [landmarks_df.loc[image_name, f'x_{19}'], landmarks_df.loc[image_name, f'y_{19}']],
            ]
        ], dtype=np.int32)
        for index1 in points:
            for index2 in index1:
                if index2[0] < 0 or index2[1] < 0:
                    return None
        return points
    else:
        points = np.array([
            [
                [landmarks_df.loc[image_name, f'x_{24}'], landmarks_df.loc[image_name, f'y_{24}']],
                [landmarks_df.loc[image_name, f'x_{24}'], landmarks_df.loc[image_name, f'y_{24}'] - (
                        landmarks_df.loc[image_name, f'y_{8}'] - int(landmarks_df.loc[image_name, f'y_{33}'] * 1.3))],
                [landmarks_df.loc[image_name, f'x_{19}'], landmarks_df.loc[image_name, f'y_{19}'] - (
                        landmarks_df.loc[image_name, f'y_{8}'] - int(landmarks_df.loc[image_name, f'y_{33}'] * 1.3))],
                [landmarks_df.loc[image_name, f'x_{19}'], landmarks_df.loc[image_name, f'y_{19}']],
            ]
        ], dtype=np.int32)
        for index1 in points:
            for index2 in index1:
                if index2[0] < 0 or index2[1] < 0:
                    return None
        return points


def left_ear(image_name, landmarks_df):
    # Left Ear
    if (landmarks_df.loc[image_name, f'x_{0}'] - landmarks_df.loc[image_name, f'x_{17}']) < -5:
        points = np.array([
            [
                [landmarks_df.loc[image_name, f'x_{0}'] - (
                        landmarks_df.loc[image_name, f'x_{17}'] - landmarks_df.loc[image_name, f'x_{0}']),
                 landmarks_df.loc[image_name, f'y_{0}']],
                [landmarks_df.loc[image_name, f'x_{1}'] - (
                        landmarks_df.loc[image_name, f'x_{17}'] - landmarks_df.loc[image_name, f'x_{1}']),
                 landmarks_df.loc[image_name, f'y_{1}']],
                [landmarks_df.loc[image_name, f'x_{2}'] - (
                        landmarks_df.loc[image_name, f'x_{17}'] - landmarks_df.loc[image_name, f'x_{2}']),
                 landmarks_df.loc[image_name, f'y_{2}']],
                [landmarks_df.loc[image_name, f'x_{3}'] - (
                        landmarks_df.loc[image_name, f'x_{17}'] - landmarks_df.loc[image_name, f'x_{3}']),
                 landmarks_df.loc[image_name, f'y_{3}']],
                [landmarks_df.loc[image_name, f'x_{3}'], landmarks_df.loc[image_name, f'y_{3}']],
                [landmarks_df.loc[image_name, f'x_{2}'], landmarks_df.loc[image_name, f'y_{2}']],
                [landmarks_df.loc[image_name, f'x_{1}'], landmarks_df.loc[image_name, f'y_{1}']],
                [landmarks_df.loc[image_name, f'x_{0}'], landmarks_df.loc[image_name, f'y_{0}']]
            ]
        ], dtype=np.int32)
        for index1 in points:
            for index2 in index1:
                if index2[0] < 0 or index2[1] < 0:
                    return None
        return points
    else:
        return None


def right_ear(image_name, landmarks_df):
    # Right Ear

    if (landmarks_df.loc[image_name, f'x_{16}'] - landmarks_df.loc[image_name, f'x_{26}']) > 5:
        points = np.array([
            [
                [landmarks_df.loc[image_name, f'x_{16}'] + (
                        landmarks_df.loc[image_name, f'x_{16}'] - landmarks_df.loc[image_name, f'x_{26}']),
                 landmarks_df.loc[image_name, f'y_{16}']],
                [landmarks_df.loc[image_name, f'x_{15}'] + (
                        landmarks_df.loc[image_name, f'x_{15}'] - landmarks_df.loc[image_name, f'x_{26}']),
                 landmarks_df.loc[image_name, f'y_{15}']],
                [landmarks_df.loc[image_name, f'x_{14}'] + (
                        landmarks_df.loc[image_name, f'x_{14}'] - landmarks_df.loc[image_name, f'x_{26}']),
                 landmarks_df.loc[image_name, f'y_{14}']],
                [landmarks_df.loc[image_name, f'x_{13}'] + (
                        landmarks_df.loc[image_name, f'x_{13}'] - landmarks_df.loc[image_name, f'x_{26}']),
                 landmarks_df.loc[image_name, f'y_{13}']],
                [landmarks_df.loc[image_name, f'x_{13}'], landmarks_df.loc[image_name, f'y_{13}']],
                [landmarks_df.loc[image_name, f'x_{14}'], landmarks_df.loc[image_name, f'y_{14}']],
                [landmarks_df.loc[image_name, f'x_{15}'], landmarks_df.loc[image_name, f'y_{15}']],
                [landmarks_df.loc[image_name, f'x_{16}'], landmarks_df.loc[image_name, f'y_{16}']]
            ]
        ], dtype=np.int32)
        for index1 in points:
            for index2 in index1:
                if index2[0] < 0 or index2[1] < 0:
                    return None
        return points
    else:
        return None


def neck(image_name, landmarks_df):
    # Neck
    points = np.array([
        [
            [landmarks_df.loc[image_name, f'x_{5}'], landmarks_df.loc[image_name, f'y_{5}']],
            [landmarks_df.loc[image_name, f'x_{6}'], landmarks_df.loc[image_name, f'y_{6}']],
            [landmarks_df.loc[image_name, f'x_{7}'], landmarks_df.loc[image_name, f'y_{7}']],
            [landmarks_df.loc[image_name, f'x_{8}'], landmarks_df.loc[image_name, f'y_{8}']],
            [landmarks_df.loc[image_name, f'x_{9}'], landmarks_df.loc[image_name, f'y_{9}']],
            [landmarks_df.loc[image_name, f'x_{10}'], landmarks_df.loc[image_name, f'y_{10}']],
            [landmarks_df.loc[image_name, f'x_{11}'], landmarks_df.loc[image_name, f'y_{11}']],
            [landmarks_df.loc[image_name, f'x_{11}'], landmarks_df.loc[image_name, f'y_{11}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{10}'], landmarks_df.loc[image_name, f'y_{10}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{9}'], landmarks_df.loc[image_name, f'y_{9}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{8}'], landmarks_df.loc[image_name, f'y_{8}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{7}'], landmarks_df.loc[image_name, f'y_{7}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{6}'], landmarks_df.loc[image_name, f'y_{6}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
            [landmarks_df.loc[image_name, f'x_{5}'], landmarks_df.loc[image_name, f'y_{5}'] - (
                    landmarks_df.loc[image_name, f'y_{33}'] - landmarks_df.loc[image_name, f'y_{58}'])],
        ]
    ], dtype=np.int32)
    return points


def five_oclock_shadow(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "5_o_Clock_Shadow_mask_" + image_name, annotations_df)
    return points


def arched_eyebrows(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Arched_Eyebrows_mask_" + image_name, annotations_df)
    return points


def attractive(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

        points.append(top_of_head(image_name, landmarks_df))

        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Attractive_mask_" + image_name, annotations_df)
    return points


def bags_under_eyes(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bags_Under_Eyes_mask_" + image_name, annotations_df)
    return points


def bald(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bald_mask_" + image_name, annotations_df)
    return points


def bangs(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bangs_mask_" + image_name, annotations_df)
    return points


def big_lips(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Big_Lips_mask_" + image_name, annotations_df)
    return points


def big_nose(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Big_Nose_mask_" + image_name, annotations_df)
    return points


def black_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Black_Hair_mask_" + image_name, annotations_df)
    return points


def blond_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Blond_Hair_mask_" + image_name, annotations_df)
    return points


def blurry(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

        points.append(top_of_head(image_name, landmarks_df))

        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Blurry_mask_" + image_name, annotations_df)
    return points


def brown_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Brown_Hair_mask_" + image_name, annotations_df)
    return points


def bushy_eyebrows(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bushy_Eyebrows_mask_" + image_name, annotations_df)
    return points


def chubby(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Chubby_mask_" + image_name, annotations_df)
    return points


def double_chin(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Double_Chin_mask_" + image_name, annotations_df)
    return points


def eyeglasses(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Eyeglasses_mask_" + image_name, annotations_df)
    return points


def goatee(image_name, annotations_df, landmarks_df, column):
    points =[]
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Goatee_mask_" + image_name, annotations_df)
    return points


def grey_hair(image_name, annotations_df, landmarks_df, column):
    points =[]
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Grey_Hair_mask_" + image_name, annotations_df)
    return points


def heavy_makeup(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Heavy_Makeup_mask_" + image_name, annotations_df)
    return points


def high_cheeckbones(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "High_Cheekbones_mask_" + image_name, annotations_df)
    return points


def male(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

        points.append(top_of_head(image_name, landmarks_df))

        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Male_mask_" + image_name, annotations_df)
    return points


def mouth_slightly_open(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Mouth_Slightly_Open_mask_" + image_name, annotations_df)
    return points


def mustache(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(upper_lip(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Mustache_mask_" + image_name, annotations_df)
    return points


def narrow_eyes(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Narrow_Eyes_mask_" + image_name, annotations_df)
    return points


def no_beard(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "No_Beard_mask_" + image_name, annotations_df)
    return points


def oval_face(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Oval_Face_mask_" + image_name, annotations_df)
    return points


def pale_skin(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Pale_Skin_mask_" + image_name, annotations_df)
    return points


def pointy_nose(image_name, annotations_df, landmarks_df, column):

    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Pointy_Nose_mask_" + image_name, annotations_df)
    return points


def receding_hairline(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Receding_Hairline_mask_" + image_name, annotations_df)
    return points


def rosy_cheeks(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Rosy_Cheeks_mask_" + image_name, annotations_df)
    return points


def side_burns(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Side_Burns_mask_" + image_name, annotations_df)
    return points


def smiling(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Smiling_mask_" + image_name, annotations_df)
    return points


def straight_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Straight_Hair_mask_" + image_name, annotations_df)
    return points


def wavy_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wavy_Hair_mask_" + image_name, annotations_df)
    return points


def wearing_earrings(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Earrings_mask_" + image_name, annotations_df)
    return points


def wearing_hat(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Hat_mask_" + image_name, annotations_df)
    return points


def wearing_lipstick(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Lipstick_mask_" + image_name, annotations_df)
    return points


def wearing_necklace(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Necklace_mask_" + image_name, annotations_df)
    return points


def wearing_necktie(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Necktie_mask_" + image_name, annotations_df)
    return points


def young(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(chin(image_name, landmarks_df))

        points.append(left_cheek(image_name, landmarks_df))

        points.append(right_cheek(image_name, landmarks_df))

        points.append(upper_lip(image_name, landmarks_df))

        points.append(neck(image_name, landmarks_df))

        points.append(mouth(image_name, landmarks_df))

        points.append(left_eye(image_name, landmarks_df))

        points.append(right_eye(image_name, landmarks_df))

        points.append(nose(image_name, landmarks_df))

        points.append(left_eye_brow(image_name, landmarks_df))

        points.append(right_eye_brow(image_name, landmarks_df))

        points.append(top_of_head(image_name, landmarks_df))

        points.append(left_ear(image_name, landmarks_df))

        points.append(right_ear(image_name, landmarks_df))


    return points

def get_attribute_points(image_name, annotations, landmarks, attribute='mouth'):

    ''' Gets the points that define a given attribute'''

    attribute_functions = {
        '5_o_Clock_Shadow': five_oclock_shadow,
        'Arched_Eyebrows': arched_eyebrows,
        'Attractive': attractive,
        'Bags_Under_Eyes': bags_under_eyes,
        'Bald': bald,
        'Bangs': bangs,
        'Big_Lips': big_lips,
        'Big_Nose': big_nose,
        'Black_Hair': black_hair,
        'Blond_Hair': blond_hair,
        'Blurry': blurry,
        'Brown_Hair': brown_hair,
        'Bushy_Eyebrows': bushy_eyebrows,
        'Chubby': chubby,
        'Double_Chin': double_chin,
        'Eyeglasses': eyeglasses,
        'Goatee': goatee,
        'Gray_Hair': grey_hair,
        'Heavy_Makeup': heavy_makeup,
        'High_Cheekbones': high_cheeckbones,
        'Male': male,
        'Mouth_Slightly_Open': mouth_slightly_open,
        'Mustache': mustache,
        'Narrow_Eyes': narrow_eyes,
        'No_Beard': no_beard,
        'Oval_Face': oval_face,
        'Pale_Skin': pale_skin,
        'Pointy_Nose': pointy_nose,
        'Receding_Hairline': receding_hairline,
        'Rosy_Cheeks': rosy_cheeks,
        'Sideburns': side_burns,
        'Smiling': smiling,
        'Straight_Hair': straight_hair,
        'Wavy_Hair': wavy_hair,
        'Wearing_Earrings': wearing_earrings,
        'Wearing_Hat': wearing_hat,
        'Wearing_Lipstick': wearing_lipstick,
        'Wearing_Necklace': wearing_necklace,
        'Wearing_Necktie': wearing_necktie,
        'Young': young
    }

    attribute_points = attribute_functions[attribute](image_name, annotations, landmarks, attribute)
    return attribute_points

def create_attribute_mask(mask_image, attribute_points):
    '''Creates an attribute mask for the given image for a given attribute  '''
    for region in attribute_points:
        cv2.fillPoly(img=mask_image, pts=region, color=(255))
    return mask_image

def create_attribute_masks(input_directory='OpenFace_not_detected_originals',
                           output_directory_1='OpenFace_not_detected_masks',
                           output_directory_2='OpenFace_not_detected_masks_resized',
                           landmarks_path='OpenFace_not_detected_landmarks.csv', annotations_path='list_attr_celeba.txt',
                           path_to_crop_points='OpenFace_not_detected_crop_points.csv'):
    ''' Creates attribute masks for the images in the given directory and saves them to
    the output directory. A subdirectory is created in the output directory for every attribute
    in the attribute list'''
    image_names = sorted(os.listdir(input_directory))

    annotations_df = pd.read_csv(annotations_path, sep='\s+', header=0, index_col="image_name")
    landmarks_df = pd.read_csv(landmarks_path, index_col='image_name')
    crop_points_df = pd.read_csv(path_to_crop_points, index_col=0)

    if not os.access(output_directory_1, os.F_OK):
        os.mkdir(output_directory_1)
    if not os.access(output_directory_2, os.F_OK):
        os.mkdir(output_directory_2)

    for image_name in tqdm.tqdm(image_names):
        image_path = input_directory + '/' + image_name
        image = cv2.imread(image_path)

        # crop and resize the mask before saving
        x_min, x_max, y_min, y_max = crop_points_df.loc[image_name]
        cropped_image = image[y_min:y_max, x_min:x_max]

        if (cropped_image.shape[1] <= 178):
            scale_factor = (179 / cropped_image.shape[1])
            resized_dim = (round(cropped_image.shape[1] * scale_factor), round(cropped_image.shape[0] * scale_factor))
            cropped_image = cv2.resize(cropped_image, resized_dim)
        if (cropped_image.shape[0] <= 218):
            scale_factor = (219 / cropped_image.shape[0])
            resized_dim = (round(cropped_image.shape[1] * scale_factor), round(cropped_image.shape[0] * scale_factor))
            cropped_image = cv2.resize(cropped_image, resized_dim)

        image_resized_path = os.path.join(output_directory_2, image_name)

        for attribute in annotations_df:
            mask_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
            attribute_points = get_attribute_points(image_name, annotations_df, landmarks_df, attribute)
            create_attribute_mask(mask_image, attribute_points)

            cropped_mask = mask_image[y_min:y_max, x_min:x_max]

            if (cropped_mask.shape[1] <= 178):
                scale_factor = (179 / cropped_mask.shape[1])
                resized_dim = (round(cropped_mask.shape[1] * scale_factor), round(cropped_mask.shape[0] * scale_factor))
                cropped_mask = cv2.resize(cropped_mask, resized_dim)
            if (cropped_mask.shape[0] <= 218):
                scale_factor = (219 / cropped_mask.shape[0])
                resized_dim = (round(cropped_mask.shape[1] * scale_factor), round(cropped_mask.shape[0] * scale_factor))
                cropped_mask = cv2.resize(cropped_mask, resized_dim)

            # save it
            mask_name = attribute + '_mask_' + image_name
            mask_path = os.path.join(output_directory_1, mask_name)

            if cv2.imwrite(mask_path, cropped_mask) == False:
                print(f'Mask {mask_name} not saved successfully!')

        if cv2.imwrite(image_resized_path, cropped_image) == False:
            print(f'Image {image_resized_path} not saved successfully!')

        # cv2.imshow(image_name + " Resized", cropped_image)
        # cv2.imshow(image_name + " " + attribute, cropped_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# attribute_list = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips',
#                   'Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin',
#                   'Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
#                   'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline',
#                   'Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat',
#                   'Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']

# path_to_landmarked_images = "/Users/nthom/Development/datasets/celeba/img_celeba/"
path_to_original_images = "/mnt/nvme0n1p1/facial_segmentation/img_celeba"
path_to_mask_images = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_masks_resized"
path_to_resized_images = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_resized"

# path_to_mask_images = "/mnt/nvme0n1p1/img_celeba_masks_resized"
# path_to_resized_images = "/mnt/nvme0n1p1/img_celeba_resized"

path_to_landmark_csv = "/mnt/nvme0n1p1/facial_segmentation/complete_unaligned_landmarks.csv"
path_to_crop_points_csv = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_crop_points.csv"
path_to_image_annotations = "/mnt/nvme0n1p1/facial_segmentation/list_attr_celeba.txt"

create_attribute_masks(path_to_original_images, path_to_mask_images, path_to_resized_images, path_to_landmark_csv,
                       path_to_image_annotations, path_to_crop_points_csv)
