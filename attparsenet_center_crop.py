import torch
import cv2

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from kornia import geometry

class AttParseNetCenterCrop(object):
    """Crop randomly the image and masks in a sample
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self, output_size, mask_output_size, segment_flag, evaluating_flag):
        # Ensure that the arguments passed in are of the expected format
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        assert isinstance(mask_output_size, (int, tuple))
        if isinstance(mask_output_size, int):
            self.mask_output_size = (mask_output_size, mask_output_size)
        else:
            assert len(mask_output_size) == 2
            self.mask_output_size = mask_output_size

        assert isinstance(segment_flag, (bool))
        self.segment_flag = segment_flag
        assert isinstance(evaluating_flag, (bool))
        self.evaluating_flag = evaluating_flag

    def __call__(self, sample):
        image = sample["image"]

        image = torch.squeeze(geometry.center_crop(torch.unsqueeze(image, 0), self.output_size, interpolation='nearest'))

        if self.segment_flag == True and self.evaluating_flag == False:
            masks = sample["masks"]

            masks = torch.squeeze(geometry.center_crop(torch.unsqueeze(masks, 0), self.output_size, interpolation='nearest'))

            # Return the randomly cropped image and masks, note that attributes were not transformed
            return {'image': image, 'attributes': sample['attributes'], 'masks': masks}

        # if ((args.test == True or args.validate == True) and args.train_by_num_epoch == False):
        #     top2 = torch.randint(0, image_h - new_image_h, (1,))
        #     left2 = torch.randint(0, image_w - new_image_w, (1,))
        #
        #     top3 = torch.randint(0, image_h - new_image_h, (1,))
        #     left3 = torch.randint(0, image_w - new_image_w, (1,))
        #
        #     image2 = image.narrow(1, top[0], new_image_h)
        #     image2 = image2.narrow(2, left[0], new_image_w)
        #
        #     image3 = image.narrow(1, top[0], new_image_h)
        #     image3 = image3.narrow(2, left[0], new_image_w)
        #
        #     image_1 = image1.shape
        #     image_2 = image2.shape
        #     image_3 = image3.shape
        #
        #     return {'image': image1, 'image2': image2, 'image3': image3, 'attributes': sample['attributes']}

        temp = image.shape
        return {'image': image, 'attributes': sample['attributes']}