import attparsenet_utils
import os
import torch
import time
import cv2
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models

# from skimage import transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

# Get args from attparsenet_utils.py
args = attparsenet_utils.get_args()
activation = None

class AttParseNetRandomCrop(object):
    """Crop randomly the image and masks in a sample. The masks are then resized to dimensions of 21x21

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, mask_output_size):
        # Ensure that the arguments passed in are of the expected format
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.mask_output_size = mask_output_size

    def __call__(self, sample):
        image = sample["image"]

        # Uncommnet the next three lines for testing: input image before resizing
        # image = image.numpy()
        # image = image.transpose(1, 2, 0)
        # cv2.imwrite("image_test.png", image)

        # Get the height and width of the image in sample
        image_h, image_w = image.shape[1:3]

        # Copy the height and width from output size
        new_image_h, new_image_w = self.output_size

        # Randomly select a point to crop the top and left edge of the image to
        top = torch.randint(0, image_h - new_image_h, (1,))
        left = torch.randint(0, image_w - new_image_w, (1,))
        # top = np.random.randint(0, image_h - new_image_h)
        # left = np.random.randint(0, image_w - new_image_w)

        # Narrow (crop) the image
        image = image.narrow(1, top[0], new_image_h)
        image = image.narrow(2, left[0], new_image_w)

        if args.segment == True:
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
                mask_np = cv2.resize(mask_np, (self.mask_output_size[1], self.mask_output_size[0]))

                # Ensure that all pixel values are either 0 or 255
                mask_np = (mask_np > 0).astype(np.uint8) * 255

                # Convert back to tensor
                mask_np = TF.to_tensor(mask_np)

                # Reconstruct the tensors so that they have a dimension of 40x96x76
                if output_masks is None:
                    output_masks = mask_np
                else:
                    output_masks = torch.cat((mask_np, output_masks))

            # Return the randomly cropped image and masks, note that attributes were not transformed
            return {'image': image, 'attributes': sample['attributes'], 'masks': output_masks}

        return {'image': image, 'attributes': sample['attributes']}


class AttParseNetDataset(Dataset):
    """AttParseNet dataset."""

    def __init__(self, args, transform=None):
        # Save the custom arguments, which are retrieved from the attparsenet_utils.py file
        self.args = args

        # Read the binary attribute labels from the specified file
        self.attr_labels = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])

        # Get the paths to each of the input images
        self.input_filenames = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[0])

        # Get the paths to each of the segment label images (masks)
        self.mask_label_filenames = pd.read_csv(args.mask_label_path, sep=',', usecols=[n for n in range(2, 42)])

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get full path to the current input image
        img_name = os.path.join(self.args.image_path, self.input_filenames.iloc[idx, 0])

        # Read the image into memory, convert it to numpy, create a copy of it, and finally make the image a tensor
        # read_image = Image.open(img_name)
        # np_image = np.asarray(read_image).transpose(2, 0, 1)
        # image = np.copy(np_image)
        # image = torch.from_numpy(image)
        image = cv2.imread(img_name)
        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image)

        # Read in the attribute labels for the current input image
        attributes = self.attr_labels.iloc[idx,]
        attributes = np.asarray(attributes)
        # Iterate over each of the input images and change and "-1" labels to "0"
        for index in range(len(attributes)):
            if attributes[index] == -1:
                attributes[index] = 0
        # Convert the labels to floats, I think that this was necessary for training
        attributes = torch.from_numpy(attributes).float()

        if args.segment == False:
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
            mask = cv2.imread(os.path.join(self.args.mask_image_path, filename), 0)
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

# Define the CNN architecture
class AttParseNet(nn.Module):
    def __init__(self):
        super(AttParseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 75, (7, 7))
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(75, 200, (3, 3))
        self.conv3 = nn.Conv2d(200, 300, (3, 3))
        self.conv4 = nn.Conv2d(300, 512, (3, 3))
        self.conv5 = nn.Conv2d(512, 512, (3, 3))
        self.conv6 = nn.Conv2d(512, 40, (3, 3))
        self.fc1 = nn.Linear(40 * 96 * 76, 40)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x_maps = x
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x, x_maps

def show_batch(batch, show_input=True, show_masks=True):
    images_batch, attributes_batch, masks_batch = batch['image'], \
                                                  batch['attributes'], \
                                                  batch['masks']

    ########## SHOW INPUT ##########
    if show_input:
        for index, image in enumerate(images_batch):
            image = image.numpy()
            image = image.transpose(1, 2, 0)
            # Note that there are two ways to view an image. Save the image and open it, or
            #      show the image while the program is running. Either uncomment imshow and waitKey
            #      or imwrite
            # cv2.imshow("Input Image", image)
            # cv2.waitKey(0)
            cv2.imwrite(f"batch_image_{index}.png", image)

    ########## SHOW MASKS ##########
    if show_masks:
        for index1, masks in enumerate(masks_batch):
            for index2, image in enumerate(masks):
                image = image.numpy()
                # Note that there are two ways to view an image. Save the image and open it, or
                #      show the image while the program is running. Either uncomment imshow and
                #      waitKey or imwrite
                # plt.imshow(image, cmap="gray")
                # plt.show()
                cv2.imwrite(f"batch_mask_{index1}_{index2}.png", image)
    ####################


def compute_metric_counts(attribute_preds, attribute_labels, true_pos_count, true_neg_count, false_pos_count, false_neg_count):
    # We use the BCEWithLogits loss function, so the sigmoid needs to be applied before computing our metrics
    attribute_preds = torch.sigmoid(attribute_preds)
    attribute_preds = torch.round(attribute_preds)

    # Remove the predictions from GPU and move to CPU
    attribute_preds_cpu = attribute_preds.detach().to(torch.device("cpu"))
    attribute_labels_cpu = attribute_labels.detach().to(torch.device("cpu"))

    attribute_positive_preds = torch.ge(attribute_preds_cpu, 1)
    attribute_negative_preds = torch.lt(attribute_positive_preds, 1)
    attribute_positive_labels = torch.ge(attribute_labels_cpu, 1)
    attribute_negative_labels = torch.lt(attribute_positive_labels, 1)

    true_positive = torch.sum((attribute_positive_preds & attribute_positive_labels).int(), dim=0)
    false_positive = torch.sum((attribute_positive_preds & attribute_negative_labels).int(), dim=0)
    true_negative = torch.sum((attribute_negative_preds & attribute_negative_labels).int(), dim=0)
    false_negative = torch.sum((attribute_negative_preds & attribute_positive_labels).int(), dim=0)

    true_pos_count += true_positive
    true_neg_count += true_negative
    false_pos_count += false_positive
    false_neg_count += false_negative

# Computes total accuracy, accuracy of positive samples, accuracy of negative samples, precision, and recall of the
# current batch
def compute_metrics(true_pos_count, true_neg_count, false_pos_count, false_neg_count):
    precision = (true_pos_count / (true_pos_count + false_pos_count))
    recall = (true_pos_count / (true_pos_count + false_neg_count))
    f1 = 2 * ((precision * recall)/ (precision + recall))
    accuracy = ((true_pos_count + true_neg_count) / (
            true_pos_count + true_neg_count + false_pos_count + false_neg_count))
    accuracy_pos = (true_pos_count / (true_pos_count + false_neg_count))
    accuracy_neg = (true_neg_count / (true_neg_count + false_pos_count))

    precision = torch.where(torch.isnan(precision), torch.zeros_like(precision), precision)
    recall = torch.where(torch.isnan(recall), torch.zeros_like(recall), recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    accuracy = torch.where(torch.isnan(accuracy), torch.zeros_like(accuracy), accuracy)
    accuracy_pos = torch.where(torch.isnan(accuracy_pos), torch.zeros_like(accuracy_pos), accuracy_pos)
    accuracy_neg = torch.where(torch.isnan(accuracy_neg), torch.zeros_like(accuracy_neg), accuracy_neg)

    return precision, recall, f1, accuracy, accuracy_pos, accuracy_neg

# Function for outputting metrics. Can output to console, csv, or txt (note comma separated) file
def output(args, output_preds, start_time, accuracy, accuracy_pos, accuracy_neg, precision, recall, f1_score, test_flag,
           console=True, file=False, csv=False):
    ### TO FILE ###
    if file == True:
        fout = open(args.metrics_output_path + f"model_{args.model}_data_{args.image_path[42:]}_{int(args.segment)}_segment_{int(args.balance)}_balance_test_{test_flag}_model_{args.load_file[1:]}.txt", "w+")
        fout.write("{0:29} {1:13} {2:13} {3:13} {4:13} {5:13} {6:13}\n".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                        "Precision", "Recall", "F1"))
        fout.write('-' * 103)
        fout.write("\n")

        for attr, acc, acc_pos, acc_neg, prec, rec, f1 in zip(args.attr_list, accuracy, accuracy_pos, accuracy_neg,
                                                          precision, recall, f1_score):
            fout.write(
                "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format(attr, acc, acc_pos,
                                                                                              acc_neg, prec, rec, f1))
        fout.write('-' * 103)
        fout.write('\n')

        fout.write("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format('', torch.mean(accuracy),
                                                                                       torch.mean(accuracy_pos),
                                                                                       torch.mean(accuracy_neg),
                                                                                       torch.mean(precision),
                                                                                       torch.mean(recall),
                                                                                       torch.mean(f1_score)))

        fout.write(f"Total time: {time.time() - start_time}\n")
        fout.close()
    ##############

    ### TO CONSOLE ###
    if console == True:
        print(
            "{0:29} {1:13} {2:13} {3:13} {4:13} {5:13} {6:13}".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                      "Precision", "Recall", "F1"))
        print('-' * 103)

        for attr, acc, acc_pos, acc_neg, prec, rec, f1 in zip(args.attr_list, accuracy, accuracy_pos, accuracy_neg,
                                                          precision, recall, f1_score):
            print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}".format(attr, acc, acc_pos,
                                                                                              acc_neg, prec, rec, f1))
        print('-' * 103)
        print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}".format('', torch.mean(accuracy),
                                                                                torch.mean(accuracy_pos),
                                                                                torch.mean(accuracy_neg),
                                                                                torch.mean(precision),
                                                                                torch.mean(recall),
                                                                                torch.mean(f1_score)))

        print(f"Total time: {time.time() - start_time}")
    #################

    ### TO CSV ###
    if csv == True:
        output_preds.append(accuracy.tolist())
        output_preds.append(accuracy_pos.tolist())
        output_preds.append(accuracy_neg.tolist())
        output_preds.append(precision.tolist())
        output_preds.append(recall.tolist())
        output_preds.append(f1_score.tolist())
        output_preds_df = pd.DataFrame(output_preds)
        output_preds_df.to_csv(args.metrics_csv_output_path + f"model_{args.model}_data_{args.image_path[42:]}_{int(args.segment)}_segment_{int(args.balance)}_balance_test_{test_flag}_model_{args.load_file[1:]}.csv", sep=',')
    ##############

# Calculates a weight for each attribute in each sample such that batches are balanced to a target distribution
def balance(args, target_distribution, data):
    start_time = time.time()
    weight_matrix = torch.zeros_like(data)
    temp_pos = torch.ones(data.shape, dtype=data.dtype, layout=data.layout, device=data.device)
    temp_neg = torch.ones(data.shape, dtype=data.dtype, layout=data.layout, device=data.device)

    pos_counts = data.sum(0)
    neg_counts = (data == 0).sum(0).float()

    xp, yp = torch.where(data == 1)
    xn, yn = torch.where(data == 0)

    pos_dist = pos_counts / args.batch_size
    neg_dist = neg_counts / args.batch_size

    pos_w = (neg_dist * target_distribution) / (pos_dist * target_distribution)
    neg_w = (pos_dist * target_distribution) / (neg_dist * target_distribution)

    pos_dist = pos_dist.repeat(data.shape[0], 1)
    neg_dist = neg_dist.repeat(data.shape[0], 1)

    x1, y1 = torch.where(target_distribution <= pos_dist)
    temp_pos[x1, y1] = pos_w.repeat(data.shape[0], 1)[x1, y1]

    x2, y2 = torch.where(target_distribution <= neg_dist)
    temp_neg[x2, y2] = neg_w.repeat(data.shape[0], 1)[x2, y2]

    weight_matrix[xp, yp] = temp_pos[xp, yp]
    weight_matrix[xn, yn] = temp_neg[xn, yn]

    return weight_matrix

# Trains the model for one epoch
def train(args, net, criterion1, criterion2, optimizer, data_loader, start_time):
    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_mse_loss = 0.0
    running_iteration_time = 0.0

    if args.model == "moon" and args.segment == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda")
    net.to(device)

    # Iterate over all batches in an epoch
    for iteration_index, sample_batched in enumerate(data_loader):
        iteration_time = time.time()

        # Get the input images, attribute labels, and masks for each sample in the batch
        # Place the model and batch of data onto the GPU
        if args.segment == True:
            inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], sample_batched['masks']
            inputs, attribute_labels, mask_labels = inputs.to(device), attribute_labels.to(device), mask_labels.to(device)
        else:
            inputs, attribute_labels = sample_batched['image'], sample_batched['attributes']
            inputs, attribute_labels = inputs.to(device), attribute_labels.to(device)

        # show_batch(sample_batched, show_input=True, show_masks=True)

        # Generate batch balancing weights
        if args.balance == True:
            batch_weights = balance(args, target_distribution=0.5, data=attribute_labels)
            batch_weights = batch_weights.to(device)
            criterion1 = nn.BCEWithLogitsLoss(weight=batch_weights)

        # Zero the gradients
        net.zero_grad()

        # Get attribute and mask predictions for each sample in the batch
        if args.segment == True and args.model == "moon":
            attribute_preds = net(inputs)
            mask_preds = activation
        elif args.segment == True:
            attribute_preds, mask_preds = net(inputs)
        else:
            attribute_preds = net(inputs)

        # Compute loss for attribute prediction and segmentation separately, then add them together
        bce_loss = criterion1(attribute_preds, attribute_labels)

        if args.segment == True:
            mse_loss = criterion2(mask_preds, mask_labels)
            running_mse_loss += mse_loss.item()
            loss = bce_loss + mse_loss
        else:
            running_mse_loss += 0
            loss = bce_loss

        # Calculate backprop and step
        loss.backward()
        optimizer.step()

        running_bce_loss += bce_loss.item()
        running_total_loss += loss.item()
        running_iteration_time += time.time() - iteration_time

        if iteration_index % 10 == 0:
            print(f"Iteration {iteration_index}")
            print(f"Total loss: {running_total_loss / (iteration_index + 1)}")
            print(f"BCE loss: {running_bce_loss / (iteration_index + 1)}")
            print(f"MSE loss: {running_mse_loss / (iteration_index + 1)}")
            print(f"Iteration time: {time.time() - iteration_time}")
            print(f"Average Iteration time: {running_iteration_time / (iteration_index + 1)}")
            print(f"Total time: {time.time() - start_time} \n")

        # pd.DataFrame(attribute_preds.tolist()).to_csv(f"./preds/{iteration_index}.csv")

    return running_total_loss


def test(args, net, optimizer, criterion1, criterion2, data_loader, test_flag):
    start_time = time.time()
    output_preds = []

    true_pos_count = torch.zeros(40)
    true_neg_count = torch.zeros(40)
    false_pos_count = torch.zeros(40)
    false_neg_count = torch.zeros(40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for progress_counter, sample_batched in enumerate(data_loader):
        if test_flag:
            print(f"Test Iteration: {progress_counter}")
        else:
            print(f"Validation Iteration: {progress_counter}")

        net.zero_grad()

        if args.segment == True:
            inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], sample_batched['masks']
            inputs, attribute_labels, mask_labels = inputs.to(device), attribute_labels.to(device), mask_labels.to(device)
            attribute_preds, mask_preds = net(inputs)
        else:
            inputs, attribute_labels = sample_batched['image'], sample_batched['attributes']
            inputs, attribute_labels = inputs.to(device), attribute_labels.to(device)
            attribute_preds = net(inputs)

        # Uncomment the following line when computing metrics
        compute_metric_counts(attribute_preds, attribute_labels, true_pos_count, true_neg_count, false_pos_count, false_neg_count)

    precision, recall, f1, accuracy, accuracy_pos, accuracy_neg = compute_metrics(true_pos_count, true_neg_count,
                                                                                  false_pos_count, false_neg_count)

    output(args, output_preds, start_time, accuracy, accuracy_pos, accuracy_neg, precision, recall, f1, test_flag, csv=True, file=True, console=True)

def main():
    # # Get args from attparsenet_utils.py
    # args = attparsenet_update_utils.get_args()

    # Initialize the model
    if args.model == "attparsenet":
        dataset = AttParseNetDataset(args, transform=transforms.Compose([AttParseNetRandomCrop((178, 218), (76, 96))]))
        net = AttParseNet()
    elif args.model == "vgg16":
        dataset = AttParseNetDataset(args, transform=transforms.Compose([AttParseNetRandomCrop((178, 218), (76, 96))]))
        net = models.vgg16()
        net.classifier[6] = nn.Linear(4096, 40)
    elif args.model == "moon":
        # dataset = AttParseNetDataset(args, transform=transforms.Compose([AttParseNetRandomCrop((178, 218), (44, 54))]))
        dataset = AttParseNetDataset(args, transform=transforms.Compose([AttParseNetRandomCrop((178, 218), (11, 13))]))
        net = models.vgg16()
        net.classifier[6] = nn.Linear(4096, 40)
        if args.segment == True:
            net.features[28] = nn.Conv2d(512, 40, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
            net.classifier[0] = nn.Linear(1960, 4096)
            # net.features[14] = nn.Conv2d(256, 40, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
            # net.features[17] = nn.Conv2d(40, 512, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))

            def get_activation():
                def hook(model, input, output):
                    global activation
                    activation = output.detach()
                return hook
            net.features[28].register_forward_hook(get_activation())

    # Collect dataset and apply transformations


    # Establish train, validate, and test splits
    if args.shuffle:
        train_indices, val_indices, test_indices = (torch.randint(low=0, high=args.train_size, size=(args.train_size,)),
                                                    torch.randint(low=args.train_size, high=args.train_size + args.val_size, size=(args.val_size,)),
                                                    torch.randint(low=args.train_size + args.val_size, high=args.all_size, size=(args.test_size,)))
    else:
        train_indices, val_indices, test_indices = (torch.tensor(list(range(args.train_size))),
                                                   torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                   torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))

    # train_indices = list(range(10))
    # val_indices = list(range(10))

    # if args.shuffle:
        # np.random.seed(args.random_seed)
        # np.random.seed(np.random.randrange(100))
        # np.random.shuffle(train_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Compute number of parameters in the network
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(net)
    print(f"Total parameters in AttParseNet: {pytorch_total_params}")

    if args.show_parameters:
        params = list(net.parameters())
        print(len(params))
        for i in range(len(params)):
            print(params[i].size())

    ########## Parallelization ##########
    if args.parallelize:
        net = nn.DataParallel(net)
    #####################################

    # Initialize loss functions and optimizer
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ########## TRAIN BY NUM EPOCH ##########
    if args.train_by_num_epoch:
        # Load a model from disk
        if args.load is True:
            checkpoint = torch.load(
                args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            net.train()
            print("Model Loaded!")
        else:
            epoch = 0

        start_time = time.time()
        min_loss = np.inf

        loss_per_epoch = []

        # Train for the number of epochs denoted in attparsenet_utils.py
        for epoch_count in tqdm(range(args.train_epochs)):
            # Store the total loss over the training epoch
            epoch_loss = train(args, net, criterion1, criterion2, optimizer, train_loader, start_time)
            print(f"Epoch {epoch + epoch_count} Loss: {epoch_loss}")

            # if (epoch_loss / len(train_loader)) < min_loss and args.save:
            #     min_loss = epoch_loss / len(train_loader)
            #     torch.save(net.state_dict(), args.save_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance/" + f"epoch_{str(epoch)}_loss_{str(epoch_loss)}")

            # torch.save(net.state_dict(), args.save_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance/"+ f"epoch_{str(epoch)}_loss_{str(epoch_loss)}")

            torch.save({
                'epoch': epoch + epoch_count,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance/model_{args.model}_data_{args.image_path[42:]}_epoch_{str(epoch+epoch_count)}_loss_{str(epoch_loss)}")

        print("Finished Training!")
    ##########

    ########## TRAIN BY COMP W/ VALIDATION ##########
    if args.train_by_comparison_with_validation:
        # Load a model from disk
        if args.load is True:
            checkpoint = torch.load(
                args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            net.train()
            print("Model Loaded!")
        else:
            epoch = 0

        start_time = time.time()
        min_loss = np.inf

        num_epochs = 0
        loss_per_epoch = []

        training_loss = np.inf
        validation_loss = np.inf

        # Continue training until the training loss is no longer greater than the validation loss
        while training_loss >= validation_loss:
            num_epochs += 1

            training_loss = train(args, net, criterion1, criterion2, optimizer, train_loader, start_time)
            validation_loss = test(args, net, optimizer, criterion1, criterion2, val_loader)

            print(f"Epoch {epoch + num_epochs} Train Loss: {training_loss}, Val Loss: {validation_loss}")

            # if args.plot_loss:
            #     loss_per_epoch.append(training_loss / len(train_loader))

            if (training_loss / len(train_loader)) < min_loss and args.save:
                min_loss = training_loss / len(train_loader)
                torch.save(net.state_dict(), args.save_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance/" + f"epoch_{str(epoch)}_loss_{str(epoch_loss)}")

            if num_epochs == 1 and validation_loss > training_loss:
                training_loss = np.inf
                validation_loss = np.inf

        # if args.plot_loss:
        #     epochs = [i for i in range(epoch + 1)]
        #     plot_loss(args, loss_per_epoch, epochs)
        print("Finished Training!")
    ###########

    if args.validate:
        # Load a model from disk
        # if args.load is True:
        #     checkpoint = torch.load(
        #         args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file)
        #     net.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     net.eval()
        #     print("Model Loaded!")

        if args.load is True:
            net.load_state_dict(torch.load(args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file), strict=False)

        test(args, net, optimizer, criterion1, criterion2, val_loader, test_flag=False)

    if args.test:
        # Load a model from disk
        # if args.load is True:
        #     checkpoint = torch.load(
        #         args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file)
        #     net.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     net.eval()
        #     print("Model Loaded!")

        if args.load is True:
            net.load_state_dict(torch.load(args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file), strict=False)

        net.load_state_dict(torch.load(args.load_path + f"{int(args.segment)}_segment_{int(args.balance)}_balance" + args.load_file), strict=False)
        test(args, net, optimizer, criterion1, criterion2, test_loader, test_flag=True)


if __name__ == "__main__":
    main()
