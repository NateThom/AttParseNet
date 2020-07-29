import utils
import os
import torch
import time
import cv2
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.autograd import Variable
from skimage import transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils as torchvision_utils
from PIL import Image


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller
        of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #Create copies of the variables that will be transformed from the values passed in
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]

	# Veryify that the variables passed in conform to the expected format
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

	# THIS SEEMS LIKE SOMETHING THAT SHOULD BE ADDRESSED IN THE README
	#     AND INCLUDED IN THE UTILS.PY FILE
        # if images are not resized before input then uncomment the next line

        # img = transform.resize(image, (new_h, new_w))

        msk = []
        # Iterate over all 40 segment labels in the masks variable
        for index in range(len(masks)):
            msk.append(transform.resize(masks[index], (new_h, new_w)))
        msk = np.asarray(msk)

        return {'image': image, 'attributes': sample['attributes'], 'masks': msk}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

	#Uncommnet the next three lines for testing: input image before resizing
        # image = image.numpy()
        # image = image.transpose(1, 2, 0)
        # cv2.imwrite("image_test.png", image)

        image_h, image_w = image.shape[1:3]

        new_image_h, new_image_w = self.output_size

        top = np.random.randint(0, image_h - new_image_h)
        left = np.random.randint(0, image_w - new_image_w)

        image = image.narrow(1, top, new_image_h)
        image = image.narrow(2, left, new_image_w)

        mask_h, mask_w = masks.shape[1:3]

        new_mask_h, new_mask_w = self.output_size

        masks = masks.narrow(1, top, new_mask_h)
        masks = masks.narrow(2, left, new_mask_w)

        output_masks = None
        for index, mask in enumerate(masks):
            mask_np = mask.numpy()
            mask_np = cv2.resize(mask_np, (76, 96))
            mask_np = (mask_np > 0).astype(np.uint8) * 255
            mask_np = TF.to_tensor(mask_np)

            if output_masks is None:
                output_masks = mask_np
            else:
                output_masks = torch.cat((mask_np, output_masks))

        return {'image': image, 'attributes': sample['attributes'], 'masks': output_masks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Currently only converts the input image."""

    def __call__(self, sample):
        image, attributes, masks = sample['image'], sample['attributes'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'attributes': torch.from_numpy(attributes),
                'masks': torch.from_numpy(masks)}


class AttParseNetDataset(Dataset):
    """AttParseNet dataset."""

    def __init__(self, args, transform=None):
        self.args = args
        self.attr_labels = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])
        self.input_filenames = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[0])
        self.mask_label_filenames = pd.read_csv(args.mask_label_path, sep=',', usecols=[n for n in range(2, 42)])
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.args.image_path, self.input_filenames.iloc[idx, 0])
        read_image = Image.open(img_name)
        np_image = np.asarray(read_image)
        image = np.copy(np_image)
        image = torch.from_numpy(image)

        attributes = self.attr_labels.iloc[idx,]
        attributes = np.asarray(attributes)
        for index in range(len(attributes)):
            if attributes[index] == -1:
                attributes[index] = 0
        attributes = torch.from_numpy(attributes).float()

        masks = None
        for filename in self.mask_label_filenames.iloc[idx,]:
            mask = cv2.imread(os.path.join(self.args.mask_image_path, filename), 0)
            if masks == None:
                masks = TF.to_tensor(mask)
            else:
                mask = TF.to_tensor(mask)
                masks = torch.cat((mask, masks))

        sample = {'image': image, 'attributes': attributes, 'masks': masks}

        if self.transform:
            print("TRANSFORM")
            sample = self.transform(sample)

        return sample


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


def plot_loss(args, loss_per_epoch, epochs):
    plt.figure()
    if loss_per_epoch[-1] > 1:
        plt.axis([0, epochs[-1], 0, max(loss_per_epoch)])
    else:
        plt.axis([0, epochs[-1], 0, 2])

    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title(f'Loss Curve - Epoch {epochs[-1] + 1}')
    plt.plot(epochs, loss_per_epoch)
    plt.savefig(args.train_plot_path + f'Pretrain_Loss_Curve_Epoch_{epochs[-1] + 1}')


def test_batch_load_time(args, data_loader):
    start_time = time.time()
    for progress_counter, batch in enumerate(data_loader):
        progress_counter += 1
        show_batch(args, batch)
        print(progress_counter)
        print(f"Total time: {time.time() - start_time}")
    print("End of test")


def compute_metrics(attribute_preds, attribute_labels, true_pos_count, true_neg_count, false_pos_count, false_neg_count):
    attribute_preds = torch.sigmoid(attribute_preds)
    attribute_preds = torch.round(attribute_preds)

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

    precision = (true_pos_count / (true_pos_count + false_pos_count))
    recall = (true_pos_count / (true_pos_count + false_neg_count))
    accuracy = ((true_pos_count + true_neg_count) / (
            true_pos_count + true_neg_count + false_pos_count + false_neg_count))
    accuracy_pos = (true_pos_count / (true_pos_count + false_neg_count))
    accuracy_neg = (true_neg_count / (true_neg_count + false_pos_count))

    precision = torch.where(torch.isnan(precision), torch.zeros_like(precision), precision)
    recall = torch.where(torch.isnan(recall), torch.zeros_like(recall), recall)
    accuracy = torch.where(torch.isnan(accuracy), torch.zeros_like(accuracy), accuracy)
    accuracy_pos = torch.where(torch.isnan(accuracy_pos), torch.zeros_like(accuracy_pos), accuracy_pos)
    accuracy_neg = torch.where(torch.isnan(accuracy_neg), torch.zeros_like(accuracy_neg), accuracy_neg)

    return precision, recall, accuracy, accuracy_pos, accuracy_neg


def output(args, output_preds, iteration_time, start_time, accuracy, accuracy_pos, accuracy_neg, precision, recall,
           console=True, file=False, csv=False):
    ### TO FILE ###
    if file == True:
        fout = open(args.metrics_output_path, "w+")
        fout.write("{0:29} {1:13} {2:13} {3:13} {4:13} {5:13}\n".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                        "Precision", "Recall"))
        fout.write('-' * 103)
        fout.write("\n")
        for attr, acc, acc_pos, acc_neg, prec, rec in zip(args.attr_list, accuracy, accuracy_pos, accuracy_neg,
                                                          precision, recall):
            fout.write(
                "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f}\n".format(attr, acc, acc_pos, acc_neg, prec,
                                                                                    rec))
        fout.write('-' * 103)
        fout.write("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f}\n".format('', torch.mean(accuracy),
                                                                                       torch.mean(accuracy_pos),
                                                                                       torch.mean(accuracy_neg),
                                                                                       torch.mean(precision),
                                                                                       torch.mean(recall)))

        fout.write(f"Iteration time: {time.time() - iteration_time}\n")
        fout.write(f"Total time: {time.time() - start_time}\n")
        fout.close()
    ##############

    ### TO CONSOLE ###
    if console == True:
        print(
            "{0:29} {1:13} {2:13} {3:13} {4:13} {5:13}".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg", "Precision",
                                                               "Recall"))
        print('-' * 103)
        for attr, acc, acc_pos, acc_neg, prec, rec in zip(args.attr_list, accuracy, accuracy_pos, accuracy_neg,
                                                          precision, recall):
            print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f}".format(attr, acc, acc_pos, acc_neg, prec,
                                                                                    rec))
        print('-' * 103)
        print("{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f}".format('', torch.mean(accuracy),
                                                                                torch.mean(accuracy_pos),
                                                                                torch.mean(accuracy_neg),
                                                                                torch.mean(precision),
                                                                                torch.mean(recall)))

        print(f"Iteration time: {time.time() - iteration_time}")
        print(f"Total time: {time.time() - start_time}")
    #################

    ### TO CSV ###
    if csv == True:
        output_preds.append(accuracy.tolist())
        output_preds.append(accuracy_pos.tolist())
        output_preds.append(accuracy_neg.tolist())
        output_preds.append(precision.tolist())
        output_preds.append(recall.tolist())
        output_preds_df = pd.DataFrame(output_preds)
        output_preds_df.to_csv(args.metrics_csv_output_path, sep=',')
    ##############


def train(net, optimizer, criterion1, criterion2, data_loader, start_time):
    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_l2_loss = 0.0

    for iteration_index, sample_batched in enumerate(data_loader):
        iteration_time = time.time()

        inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], \
                                                sample_batched['masks']

        show_batch(sample_batched, show_input=True, show_masks=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        inputs, attribute_labels, mask_labels = inputs.to(device), attribute_labels.to(device), mask_labels.to(device)

        optimizer.zero_grad()
        net.zero_grad()

        attribute_preds, mask_preds = net(inputs)

        loss1 = criterion1(attribute_preds, attribute_labels)
        loss2 = 2 * torch.sqrt(criterion2(mask_preds, mask_labels))
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        running_total_loss += loss.item()
        running_bce_loss += loss1.item()
        running_l2_loss += loss2.item()

        if iteration_index % 10 == 0:
            print(f"Iteration {iteration_index} Total loss: {running_total_loss / (iteration_index + 1)} "
                  f"BCE loss: {running_bce_loss / (iteration_index + 1)} "
                  f"L2 loss: {running_l2_loss / (iteration_index + 1)}")
            print(f"Iteration time: {time.time() - iteration_time}")
            print(f"Total time: {time.time() - start_time}")

    return running_total_loss


def test(net, optimizer, criterion1, criterion2, data_loader):
    running_total_loss = 0.0

    output_preds = []

    true_pos_count = torch.zeros(40)
    true_neg_count = torch.zeros(40)
    false_pos_count = torch.zeros(40)
    false_neg_count = torch.zeros(40)

    for progress_counter, sample_batched in enumerate(data_loader):
        print(f"Validation Iteration: {progress_counter}")
        iteration_time = time.time()
        inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], \
                                                sample_batched['masks']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net.to(device)
        inputs, attribute_labels, mask_labels = inputs.to(device), attribute_labels.to(device), mask_labels.to(device)

        optimizer.zero_grad()
        net.zero_grad()

        attribute_preds, mask_preds = net(inputs)

        loss1 = criterion1(attribute_preds, attribute_labels)
        loss2 = 2 * torch.sqrt(criterion2(mask_preds, mask_labels))
        loss = loss1 + loss2
        running_total_loss += loss

        #Uncomment the next 5 lines to view the metrics for each batch
        # precision, recall, accuracy, accuracy_pos, accuracy_neg = compute_metrics(attribute_preds, attribute_labels,
        #                                                                           true_pos_count, true_neg_count,
        #                                                                           false_pos_count, false_neg_count)
        #
        # output(args, output_preds, iteration_time, start_time, accuracy, accuracy_pos, accuracy_neg, precision, recall)

    return running_total_loss


def main():
    args = utils.get_args()

    dataset = AttParseNetDataset(args, transform=transforms.Compose([RandomCrop((218,178))]))
    train_indices, val_indices, test_indices = list(range(args.train_size)), \
                                               list(range(args.train_size, args.train_size + args.val_size)), \
                                               list(range(args.train_size + args.val_size, args.all_size))

    if args.shuffle:
        np.random.seed(args.random_seed)
        # np.random.seed(np.random.randrange(100))
        np.random.shuffle(train_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    net = AttParseNet()
    if args.load is True:
        net.load_state_dict(torch.load(args.load_path), strict=False)
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    ########## Parallelization ##########
    if args.parallelize:
        net = nn.DataParallel(net)
    #####################################

    print(net)
    print(f"Total parameters in AttParseNet: {pytorch_total_params}")

    if args.show_parameters:
        params = list(net.parameters())
        print(len(params))
        for i in range(len(params)):
            print(params[i].size())

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

########## TRAIN BY NUM EPOCH ##########
    if args.train_by_num_epoch:
        start_time = time.time()
        min_loss = np.inf

        loss_per_epoch = []

        for epoch in tqdm(range(args.train_epochs)):

            epoch_loss = train(net, optimizer, criterion1, criterion2, train_loader, start_time)

            print(f"Epoch {epoch} Loss: {epoch_loss}")

            if args.plot_loss:
                loss_per_epoch.append(epoch_loss / len(train_loader))

            if (epoch_loss / len(train_loader)) < min_loss and args.save:
                min_loss = epoch_loss / len(train_loader)
                torch.save(net.state_dict(), args.save_path + f"_{str(epoch)}_{str(epoch_loss)}")

        if args.plot_loss:
            epochs = [i for i in range(epoch + 1)]
            plot_loss(args, loss_per_epoch, epochs)
        print("Finished Training!")
##########

########## TRAIN BY COMP W/ VALIDATION ##########
    if args.train_by_comparison_with_validation:

        start_time = time.time()
        min_loss = np.inf

        num_epochs = 0
        loss_per_epoch = []

        training_loss = np.inf
        validation_loss = np.inf

        while training_loss >= validation_loss:
            num_epochs += 1

            training_loss = train(net, optimizer, criterion1, criterion2, train_loader, start_time)
            validation_loss = test(net, optimizer, criterion1, criterion2, val_loader)

            print(f"Epoch {num_epochs} Train Loss: {training_loss}, Val Loss: {validation_loss}")

            if args.plot_loss:
                loss_per_epoch.append(training_loss / len(train_loader))

            if (training_loss / len(train_loader)) < min_loss and args.save:
                min_loss = training_loss / len(train_loader)
                torch.save(net.state_dict(), args.save_path + f"_{str(epoch)}_{str(training_loss)}")

            if num_epochs == 1 and validation_loss > training_loss:
                training_loss = np.inf
                validation_loss = np.inf

        if args.plot_loss:
            epochs = [i for i in range(epoch + 1)]
            plot_loss(args, loss_per_epoch, epochs)
        print("Finished Training!")
###########

    if args.validate:
        net.load_state_dict(torch.load(args.load_path), strict=False)
        test(val_loader)

    if args.test:
        test(test_loader)


if __name__ == "__main__":
    main()
