import attparsenet_regions

# Each function in this file is responsible for collecting the various foundational regions from attparsenet_regions
#   and constructing the segments in which the various attributes occur

def five_oclock_shadow(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "5_o_Clock_Shadow_mask_" + image_name, annotations_df)
    return points


def arched_eyebrows(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Arched_Eyebrows_mask_" + image_name, annotations_df)
    return points


def attractive(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Attractive_mask_" + image_name, annotations_df)
    return points


def bags_under_eyes(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bags_Under_Eyes_mask_" + image_name, annotations_df)
    return points


def bald(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bald_mask_" + image_name, annotations_df)
    return points


def bangs(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bangs_mask_" + image_name, annotations_df)
    return points


def big_lips(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Big_Lips_mask_" + image_name, annotations_df)
    return points


def big_nose(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Big_Nose_mask_" + image_name, annotations_df)
    return points


def black_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Black_Hair_mask_" + image_name, annotations_df)
    return points


def blond_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Blond_Hair_mask_" + image_name, annotations_df)
    return points


def blurry(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Blurry_mask_" + image_name, annotations_df)
    return points


def brown_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Brown_Hair_mask_" + image_name, annotations_df)
    return points


def bushy_eyebrows(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Bushy_Eyebrows_mask_" + image_name, annotations_df)
    return points


def chubby(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Chubby_mask_" + image_name, annotations_df)
    return points


def double_chin(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Double_Chin_mask_" + image_name, annotations_df)
    return points


def eyeglasses(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Eyeglasses_mask_" + image_name, annotations_df)
    return points


def goatee(image_name, annotations_df, landmarks_df, column):
    points =[]
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Goatee_mask_" + image_name, annotations_df)
    return points


def grey_hair(image_name, annotations_df, landmarks_df, column):
    points =[]
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Grey_Hair_mask_" + image_name, annotations_df)
    return points


def heavy_makeup(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Heavy_Makeup_mask_" + image_name, annotations_df)
    return points


def high_cheeckbones(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "High_Cheekbones_mask_" + image_name, annotations_df)
    return points


def male(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Male_mask_" + image_name, annotations_df)
    return points


def mouth_slightly_open(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Mouth_Slightly_Open_mask_" + image_name, annotations_df)
    return points


def mustache(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Mustache_mask_" + image_name, annotations_df)
    return points


def narrow_eyes(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Narrow_Eyes_mask_" + image_name, annotations_df)
    return points


def no_beard(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "No_Beard_mask_" + image_name, annotations_df)
    return points


def oval_face(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Oval_Face_mask_" + image_name, annotations_df)
    return points


def pale_skin(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Pale_Skin_mask_" + image_name, annotations_df)
    return points


def pointy_nose(image_name, annotations_df, landmarks_df, column):

    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.nose(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Pointy_Nose_mask_" + image_name, annotations_df)
    return points


def receding_hairline(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Receding_Hairline_mask_" + image_name, annotations_df)
    return points


def rosy_cheeks(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Rosy_Cheeks_mask_" + image_name, annotations_df)
    return points


def side_burns(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Side_Burns_mask_" + image_name, annotations_df)
    return points


def smiling(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Smiling_mask_" + image_name, annotations_df)
    return points


def straight_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Straight_Hair_mask_" + image_name, annotations_df)
    return points


def wavy_hair(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wavy_Hair_mask_" + image_name, annotations_df)
    return points


def wearing_earrings(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Earrings_mask_" + image_name, annotations_df)
    return points


def wearing_hat(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Hat_mask_" + image_name, annotations_df)
    return points


def wearing_lipstick(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Lipstick_mask_" + image_name, annotations_df)
    return points


def wearing_necklace(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Necklace_mask_" + image_name, annotations_df)
    return points


def wearing_necktie(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.neck(image_name, landmarks_df))

    # cv2.imwrite("../region_masks/" + "Wearing_Necktie_mask_" + image_name, annotations_df)
    return points


def young(image_name, annotations_df, landmarks_df, column):
    points = []
    if annotations_df.loc[image_name, column] == 1:
        points.append(attparsenet_regions.chin(image_name, landmarks_df))

        points.append(attparsenet_regions.left_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.right_cheek(image_name, landmarks_df))

        points.append(attparsenet_regions.upper_lip(image_name, landmarks_df))

        points.append(attparsenet_regions.neck(image_name, landmarks_df))

        points.append(attparsenet_regions.mouth(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye(image_name, landmarks_df))

        points.append(attparsenet_regions.nose(image_name, landmarks_df))

        points.append(attparsenet_regions.left_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.right_eye_brow(image_name, landmarks_df))

        points.append(attparsenet_regions.top_of_head(image_name, landmarks_df))

        points.append(attparsenet_regions.left_ear(image_name, landmarks_df))

        points.append(attparsenet_regions.right_ear(image_name, landmarks_df))

    return points