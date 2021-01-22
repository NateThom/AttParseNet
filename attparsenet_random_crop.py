import torch
import cv2

import torchvision.transforms.functional as TF

class AttParseNetRandomCrop(object):
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

        # Get the height and width of the image in sample
        image_h, image_w = image.shape[1:3]

        # Copy the height and width from output size
        new_image_h, new_image_w = self.output_size

        # Randomly select a point to crop the top and left edge of the image to
        top = torch.randint(0, image_h - new_image_h, (1,))
        left = torch.randint(0, image_w - new_image_w, (1,))

        # Narrow (crop) the image
        image1 = image.narrow(1, top[0], new_image_h)
        image1 = image1.narrow(2, left[0], new_image_w)

        if self.segment_flag == True and self.evaluating_flag == False:
            masks = sample["masks"]

            # Copy the height and width from output size
            new_mask_h, new_mask_w = self.output_size

            # Narrow (crop) the masks to the same randomly selected parameters as image
            masks = masks.narrow(1, top[0], new_mask_h)
            masks = masks.narrow(2, left[0], new_mask_w)

            # Resize each of the images to dimensions of 96x76 and ensure that all pixel values are either 0 or 255
            output_masks = None
            for index, mask in enumerate(masks):
                # Convert to numpy and resize
                mask_np = mask.numpy()
                mask_np = cv2.resize(mask_np, (self.mask_output_size[1], self.mask_output_size[0]), interpolation=0)

                # Convert back to tensor
                mask_np = TF.to_tensor(mask_np)

                # Reconstruct the tensors so that they have a dimension of 40x96x76
                if output_masks is None:
                    output_masks = mask_np
                else:
                    output_masks = torch.cat((mask_np, output_masks))

            # Return the randomly cropped image and masks, note that attributes were not transformed
            return {'image': image1, 'attributes': sample['attributes'], 'masks': output_masks}

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

        return {'image': image1, 'attributes': sample['attributes']}