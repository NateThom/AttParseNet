import torch
import os
import cv2

import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset

class AttParseNetDataset(Dataset):
    """AttParseNet dataset."""

    def __init__(self, segment_flag, evaluating_flag, image_path, image_dir, mask_image_path, attr_label_path, mask_label_path, transform=None):
        assert isinstance(segment_flag, (bool))
        self.segment_flag = segment_flag
        assert isinstance(evaluating_flag, (bool))
        self.evaluating_flag = evaluating_flag
        assert isinstance(image_path, (str))
        self.image_path = image_path
        assert isinstance(image_dir, (str))
        self.image_dir = image_dir
        assert isinstance(mask_image_path, (str))
        self.mask_image_path = mask_image_path
        assert isinstance(attr_label_path, (str))
        assert isinstance(mask_label_path, (str))

        # Read the binary attribute labels from the specified file
        # self.attr_labels = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])
        self.attr_labels = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 6)])

        # Get the paths to each of the input images
        self.input_filenames = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[0])
        # Get the paths to each of the segment label images (masks)
        # self.mask_label_filenames = pd.read_csv(mask_label_path, sep=',', usecols=[n for n in range(2, 42)])
        self.mask_label_filenames = pd.read_csv(mask_label_path, sep=',', usecols=[n for n in range(2, 7)])

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get full path to the current input image
        img_name = os.path.join(self.image_path, self.image_dir, self.input_filenames.iloc[idx, 0])

        image = cv2.imread(img_name)
        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image)

        # Read in the attribute labels for the current input image
        attributes = self.attr_labels.iloc[idx,]
        # attributes = np.asarray(attributes)
        attributes = torch.tensor(attributes)
        attributes = torch.gt(attributes, 0).float()

        ## Iterate over each of the input images and change and "-1" labels to "0"
        # for index in range(len(attributes)):
        #     if attributes[index] == -1:
        #         attributes[index] = 0
        ## Convert the labels to floats, I think that this was necessary for training
        # attributes = torch.from_numpy(attributes).float()

        # if ((args.test == True or args.validate == True) and args.train_by_num_epoch == False):
        #     sample = {'image': image, 'image2': image, 'image3': image, 'attributes': attributes}
        #     if self.transform:
        #         sample = self.transform(sample)
        #
        #     image_1 = sample['image'].shape
        #     image_2 = sample['image2'].shape
        #     image_3 = sample['image3'].shape
        #     return sample
        # elif args.segment == False or ((args.test == True or args.validate == True) and args.train_by_num_epoch == False):
        #     sample = {'image': image, 'attributes': attributes}
        #     if self.transform:
        #         sample = self.transform(sample)
        #     return sample

        if self.segment_flag == False or self.evaluating_flag == True:
            sample = {'image': image, 'attributes': attributes}
            if self.transform:
                sample = self.transform(sample)
            return sample

        #mask_start_time = time.time()
        # Read all of the segment label images for the current input image

        #masks = None
        #for filename in self.mask_label_filenames.iloc[idx,]:
        #    mask = cv2.imread(os.path.join(self.args.mask_image_path, filename), 0)
        #    if masks == None:
        #        masks = TF.to_tensor(mask)
        #    else:
        #        mask = TF.to_tensor(mask)
        #        masks = torch.cat((mask, masks))

        masks = None
        temp = None
        cat_count = 0
        for filename in self.mask_label_filenames.iloc[idx,]:
            cat_count += 1
            mask = cv2.imread(os.path.join(self.mask_image_path, filename), 0)
            if masks == None:
                masks = TF.to_tensor(mask)
            elif temp == None:
                temp = TF.to_tensor(mask)
            else:
                mask = TF.to_tensor(mask)
                temp = torch.cat((mask, temp))
                if cat_count == 10:
                    masks = torch.cat((temp, masks))
                    temp = None
                    cat_count = 0
        if temp != None:
                masks = torch.cat((temp, masks))

        #mask_end_time = time.time()
        #print(mask_end_time - mask_start_time)
        #a

        sample = {'image': image, 'attributes': attributes, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample