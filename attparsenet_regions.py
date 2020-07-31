import numpy as np

# Each function in this file is responsible for forming the various foundational regions of a face from known
#   landmark points

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