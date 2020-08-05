import dlib
import os
import subprocess
import cv2
import shutil
import tqdm

import pandas as pd
import numpy as np


# gets bbox and inserts it into a dictionary
def use_bbox(path_to_celeba_bbox_csv):
    celeba_bbox_df = pd.read_csv(path_to_celeba_bbox_csv, skiprows=1, sep="\s+")

    bbox_dict = {}
    for x in celeba_bbox_df.index:
        # saves values as tuples
        value = (celeba_bbox_df.at[x, "x_1"], celeba_bbox_df.at[x, "y_1"], celeba_bbox_df.at[x, "width"],
                 celeba_bbox_df.at[x, "height"])

        # uses image name as the dictionary key
        bbox_dict[celeba_bbox_df.at[x, "image_id"]] = value

    return bbox_dict


####################
####################

# works on entire directories
def process_directory_opencv(path_to_input_images, bbox_dict, num_detected, num_not_detected):
    if not os.path.exists("/home/nthom/Documents/AttParseNet/detected_opencv"):
        os.mkdir("/home/nthom/Documents/AttParseNet/detected_opencv")
    if not os.path.exists("/home/nthom/Documents/AttParseNet/not_detected"):
        os.mkdir("/home/nthom/Documents/AttParseNet/not_detected")

    # this section creates the columns for the csv file (hardcoded to work with 68 predictor)
    temp_list = []
    for landmark_index in range(68):
        temp_list.append("x_" + str(landmark_index) + ",")
    for landmark_index in range(68):
        temp_list.append("y_" + str(landmark_index) + ",")
    df = pd.DataFrame(columns=[col for col in temp_list])

    entries = sorted(os.listdir(path_to_input_images))
    for entry in tqdm.tqdm(entries[:100]):
        img = cv2.imread(path_to_input_images + "/" + entry)
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img_gray, 1)

            if len(rects) == 0:
                num_not_detected += 1
                cv2.imwrite(os.path.join('/home/nthom/Documents/AttParseNet/not_detected/' + entry), img)

            # if a face is detected then we go through the cropping process
            else:
                num_detected += 1

                # this part identifies the features using dlib's face predictor
                rect = get_rect(rects, bbox_dict[entry])

                # here, the corners of the cropping square are identified
                shape = predictor(img_gray, rect)
                shape = shape_to_np(shape)

                # The following lines can be used for testing that the detector is finding the correct face
                # (x, y, w, h) = rect_to_bb(rect)
                # cv2.imshow("Detector Test", cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0)))

                cv2.imwrite(os.path.join('/home/nthom/Documents/AttParseNet/detected_opencv/' + entry), img)

                temp_list = []
                for dimension in range(len(shape[0])):
                    for landmark in range(len(shape)):
                        temp_list.append(shape[landmark][dimension])

                # temp_list = []
                # for landmark_coordinate in range(len(shape)):
                #     for dimension in len(range(shape[landmark_coordinate])):
                #         temp_list.append(shape[landmark_coordinate[dimension]])

                # appends x and y coordinates as a single tuple
                df_length = len(df)
                df.loc[df_length] = temp_list
                df.index = df.index[:-1].tolist() + [entry]

    return df, detected, not_detected


def process_directory_openface(path_to_input_images, bbox_dict, num_detected, num_not_detected):
    if not os.path.exists("/home/nthom/Documents/AttParseNet/OpenFace_detected"):
        os.mkdir("/home/nthom/Documents/AttParseNet/OpenFace_detected")
    if not os.path.exists("/home/nthom/Documents/AttParseNet/OpenFace_not_detected"):
        os.mkdir("/home/nthom/Documents/AttParseNet/OpenFace_not_detected")
    if not os.path.exists("/home/nthom/Documents/AttParseNet/OpenFace_landmarks"):
        os.mkdir("/home/nthom/Documents/AttParseNet/OpenFace_landmarks")

    open_face_bash_command = "/home/nthom/Documents/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir " + path_to_input_images + " -out_dir /home/nthom/Documents/AttParseNet/OpenFace_landmarks"
    subprocess.call(open_face_bash_command.split())

    # this section creates the columns for the csv file (hardcoded to work with 68 predictor)
    temp_list = []
    for landmark_index in range(68):
        temp_list.append("x_" + str(landmark_index) + ",")
    for landmark_index in range(68):
        temp_list.append("y_" + str(landmark_index) + ",")
    landmark_df = pd.DataFrame(columns=[col for col in temp_list])

    entries = sorted(os.listdir(path_to_input_images))
    for entry in entries:
        landmark_df = extract_landmarks_openface(entry, '/home/nthom/Documents/AttParseNet/OpenFace_landmarks/', landmark_df, path_to_input_images,
                                                 bbox_dict, '/home/nthom/Documents/AttParseNet/OpenFace_detected', num_detected, num_not_detected)

    return landmark_df


def get_rect(rects, bbox):
    # print(len(rects))
    if (len(rects) == 1):
        return rects[0]
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1, y1))
        dist = float("inf")
        closest_bbox = None
        #  print(rects)
        for i, rect in enumerate(rects):
            # print(rect)
            (x, y, w, h) = rect_to_bb(rect)
            dlib_bbox = np.array((x, y))
            # print(celebA_bbox - dlib_bbox)
            if np.linalg.norm(celebA_bbox - dlib_bbox) < dist:
                closest_bbox = rect
                dist = np.linalg.norm(celebA_bbox - dlib_bbox)
        # print(closest_bbox)
        return closest_bbox


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


# imported from face utils
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def extract_landmarks_openface(image_name, openface_output_directory, landmark_df, path_to_input_images, bbox_dict,
                               output_directory, num_detected, num_not_detected):
    entry = openface_output_directory + image_name[:-4] + ".csv"
    file_path = path_to_input_images + '/' + image_name

    # Check if landmark csv exists for the current input image
    if os.path.isfile(entry) is False:
        num_not_detected += 1
        shutil.copy(file_path, "/home/nthom/Documents/AttParseNet/OpenFace_not_detected")
        return landmark_df
    # If the landmark csv does exist
    else:
        num_detected += 1
        img = cv2.imread(file_path)
        temp_list = []
        old_df = pd.read_csv(entry)
        k = get_rect_OpenFace(old_df, bbox_dict)
        for i in range(68):
            value = (old_df.at[k, " x_" + str(i)], old_df.at[k, " y_" + str(i)])
            temp_list.append(value)

        df_length = len(landmark_df)
        landmark_df.loc[df_length] = temp_list
        landmark_df.index = landmark_df.index[:-1].tolist() + [image_name]

        cv2.imwrite(output_directory + "/" + image_name, img)
        return landmark_df


def get_rect_OpenFace(of_landmarks, bbox):
    if len(of_landmarks.index) == 1:
        return 0
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1, y1))
        dist = float("inf")
        closest_bbox = None

        for i in of_landmarks.index:
            coords = np.array(((int(round(of_landmarks.iloc[i][2]))), (int(round(of_landmarks.iloc[i][70])))))

            if np.linalg.norm(celebA_bbox - coords) < dist:
                closest_bbox = i
                dist = np.linalg.norm(celebA_bbox - coords)
        return closest_bbox


####################
####################

def create_new_csv(df, csv_file):
    old_df = pd.read_csv(csv_file)
    labels = old_df.columns.values

    new_df = pd.DataFrame(columns=labels)

    for i in range(0, len(df)):
        name = df.index[i]
        name = int(name[:-4]) - 1
        row = old_df.loc[name]
        new_df.loc[name] = row

    return new_df


########################################
########################################

if __name__ == "__main__":
    detected = 0
    not_detected = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    bbox_dict = use_bbox("~/mnt/mworks03/home/nthom/Documents/datasets/celebA/Anno/list_bbox_celeba.txt")
    path_to_input_images = "/home/nthom/Documents/datasets/celebA/images/img_celeba_sample/"

    # opencv_df, detected, not_detected = process_directory_opencv(path_to_input_images, bbox_dict, detected, not_detected)
    # opencv_df.to_csv("opencv_landmarks.csv")
    # opencv_features = create_new_csv(opencv_df, "/home/nthom/Documents/datasets/celebA/Anno/list_attr_celeba.txt")

    openface_df, detected, not_detected = process_directory_openface(path_to_input_images, bbox_dict, detected, not_detected)
    openface_df.to_csv("openface_landmarks.csv")
    openface_features = create_new_csv(openface_df, "~/mnt/mworks03/home/nthom/Documents/datasets/celebA/Anno/list_attr_celeba.txt")

    print(f"Number of images detected {detected}")
    print(f"Number of images NOT detected: {not_detected}")
    print(f"Percentage of dataset landmarked: {detected/(detected + not_detected)}")
