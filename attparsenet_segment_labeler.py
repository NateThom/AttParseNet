import os
import cv2
import tqdm
import attparsenet_segments

import numpy as np
import pandas as pd


def get_attribute_points(image_name, annotations, landmarks, attribute='mouth'):

    ''' Gets the points that define a given attribute'''

    attribute_functions = {
        '5_o_Clock_Shadow': attparsenet_segments.five_oclock_shadow,
        'Arched_Eyebrows': attparsenet_segments.arched_eyebrows,
        'Attractive': attparsenet_segments.attractive,
        'Bags_Under_Eyes': attparsenet_segments.bags_under_eyes,
        'Bald': attparsenet_segments.bald,
        'Bangs': attparsenet_segments.bangs,
        'Big_Lips': attparsenet_segments.big_lips,
        'Big_Nose': attparsenet_segments.big_nose,
        'Black_Hair': attparsenet_segments.black_hair,
        'Blond_Hair': attparsenet_segments.blond_hair,
        'Blurry': attparsenet_segments.blurry,
        'Brown_Hair': attparsenet_segments.brown_hair,
        'Bushy_Eyebrows': attparsenet_segments.bushy_eyebrows,
        'Chubby': attparsenet_segments.chubby,
        'Double_Chin': attparsenet_segments.double_chin,
        'Eyeglasses': attparsenet_segments.eyeglasses,
        'Goatee': attparsenet_segments.goatee,
        'Gray_Hair': attparsenet_segments.grey_hair,
        'Heavy_Makeup': attparsenet_segments.heavy_makeup,
        'High_Cheekbones': attparsenet_segments.high_cheeckbones,
        'Male': attparsenet_segments.male,
        'Mouth_Slightly_Open': attparsenet_segments.mouth_slightly_open,
        'Mustache': attparsenet_segments.mustache,
        'Narrow_Eyes': attparsenet_segments.narrow_eyes,
        'No_Beard': attparsenet_segments.no_beard,
        'Oval_Face': attparsenet_segments.oval_face,
        'Pale_Skin': attparsenet_segments.pale_skin,
        'Pointy_Nose': attparsenet_segments.pointy_nose,
        'Receding_Hairline': attparsenet_segments.receding_hairline,
        'Rosy_Cheeks': attparsenet_segments.rosy_cheeks,
        'Sideburns': attparsenet_segments.side_burns,
        'Smiling': attparsenet_segments.smiling,
        'Straight_Hair': attparsenet_segments.straight_hair,
        'Wavy_Hair': attparsenet_segments.wavy_hair,
        'Wearing_Earrings': attparsenet_segments.wearing_earrings,
        'Wearing_Hat': attparsenet_segments.wearing_hat,
        'Wearing_Lipstick': attparsenet_segments.wearing_lipstick,
        'Wearing_Necklace': attparsenet_segments.wearing_necklace,
        'Wearing_Necktie': attparsenet_segments.wearing_necktie,
        'Young': attparsenet_segments.young
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

    # Get the name of each file in the input directory
    image_names = sorted(os.listdir(input_directory))

    # Open the attribute label, landmark, and crop point csv files
    annotations_df = pd.read_csv(annotations_path, sep='\s+', header=0, index_col="image_name")
    landmarks_df = pd.read_csv(landmarks_path, index_col='image_name')
    crop_points_df = pd.read_csv(path_to_crop_points, index_col=0)

    # Ensure that the output directories exist. If they do not exist, create them
    if not os.access(output_directory_1, os.F_OK):
        os.mkdir(output_directory_1)
    if not os.access(output_directory_2, os.F_OK):
        os.mkdir(output_directory_2)

    # Iterate through each image in the input directory
    for image_name in tqdm.tqdm(image_names):

        # store the complete path to the image and open it
        image_path = input_directory + '/' + image_name
        image = cv2.imread(image_path)

        # crop the image according to the crop points csv
        x_min, x_max, y_min, y_max = crop_points_df.loc[image_name]
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Ensure that the resulting image is of the correct size
        #      The image will have minimum dimensions of 179x219
        if (cropped_image.shape[1] <= 178):
            scale_factor = (179 / cropped_image.shape[1])
            resized_dim = (round(cropped_image.shape[1] * scale_factor), round(cropped_image.shape[0] * scale_factor))
            cropped_image = cv2.resize(cropped_image, resized_dim)
        if (cropped_image.shape[0] <= 218):
            scale_factor = (219 / cropped_image.shape[0])
            resized_dim = (round(cropped_image.shape[1] * scale_factor), round(cropped_image.shape[0] * scale_factor))
            cropped_image = cv2.resize(cropped_image, resized_dim)

        # Store the complete output path for the resized image
        image_resized_path = os.path.join(output_directory_2, image_name)

        # For each attribute
        for attribute in annotations_df:
            # Generate a black image with same dimensions as the resized input image
            mask_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

            # Get the associated landmark points for the current attribute
            attribute_points = get_attribute_points(image_name, annotations_df, landmarks_df, attribute)

            # Fill in the regions of the mask image, where the current attribute occurs, with white pixels
            create_attribute_mask(mask_image, attribute_points)

            # Crop the mask image
            cropped_mask = mask_image[y_min:y_max, x_min:x_max]

            if (cropped_mask.shape[1] <= 178):
                scale_factor = (179 / cropped_mask.shape[1])
                resized_dim = (round(cropped_mask.shape[1] * scale_factor), round(cropped_mask.shape[0] * scale_factor))
                cropped_mask = cv2.resize(cropped_mask, resized_dim)
            if (cropped_mask.shape[0] <= 218):
                scale_factor = (219 / cropped_mask.shape[0])
                resized_dim = (round(cropped_mask.shape[1] * scale_factor), round(cropped_mask.shape[0] * scale_factor))
                cropped_mask = cv2.resize(cropped_mask, resized_dim)

            # Store the full path to the mask output directory
            mask_name = attribute + '_mask_' + image_name
            mask_path = os.path.join(output_directory_1, mask_name)

            # Save the mask for the current attribute
            if cv2.imwrite(mask_path, cropped_mask) == False:
                print(f'Mask {mask_name} not saved successfully!')

        # Save the resized image
        if cv2.imwrite(image_resized_path, cropped_image) == False:
            print(f'Image {image_resized_path} not saved successfully!')

        # Uncomment the four lines below and comment the lines above (the ones that save the images) to see the masks
        #   and resized images

        # cv2.imshow(image_name + " Resized", cropped_image)
        # cv2.imshow(image_name + " " + attribute, cropped_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

path_to_original_images = "/mnt/nvme0n1p1/facial_segmentation/img_celeba"
path_to_output_segment_labels = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_masks_resized"
path_to_output_resized_images = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_resized"

path_to_landmark_csv = "/mnt/nvme0n1p1/facial_segmentation/complete_unaligned_landmarks.csv"
path_to_crop_points_csv = "/mnt/nvme0n1p1/facial_segmentation/img_celeba_crop_points.csv"
path_to_image_annotations = "/mnt/nvme0n1p1/facial_segmentation/list_attr_celeba.txt"

create_attribute_masks(path_to_original_images, path_to_output_segment_labels, path_to_output_resized_images,
                       path_to_landmark_csv,path_to_image_annotations, path_to_crop_points_csv)
