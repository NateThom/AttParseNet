import torch
import os

from torch import nn

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

class AttParseNet(pl.LightningModule):
    def __init__(self, hparams):
        super(AttParseNet, self).__init__()

        assert isinstance(hparams.segment, (bool))
        self.segment_flag = hparams.segment
        assert isinstance(hparams.show_batch, (bool))
        self.show_batch_flag = hparams.show_batch
        assert isinstance(hparams.repair_labels, (bool))
        self.repair_labels_flag = hparams.repair_labels
        assert isinstance(hparams.save_feature_maps, (bool))
        self.save_feature_map_flag = hparams.save_feature_maps
        assert isinstance(hparams.lr, (float))
        self.lr = hparams.lr
        assert isinstance(hparams.patience, (int))
        self.patience = hparams.patience

        self.attr_list = hparams.attr_list

        self.save_hyperparameters()

        # self.learning_rate = learning_rate

        self.train_accuracy = pl.metrics.Accuracy()
        self.train_precision = pl.metrics.Precision(num_classes=40)
        self.train_recall = pl.metrics.Recall(num_classes=40)
        self.train_f1 = pl.metrics.F1(num_classes=40)

        self.valid_accuracy = pl.metrics.Accuracy()
        self.valid_precision = pl.metrics.Precision(num_classes=40)
        self.valid_recall = pl.metrics.Recall(num_classes=40)
        self.valid_f1 = pl.metrics.F1(num_classes=40)

        self.test_accuracy = pl.metrics.Accuracy()
        self.test_precision = pl.metrics.Precision(num_classes=40)
        self.test_recall = pl.metrics.Recall(num_classes=40)
        self.test_f1 = pl.metrics.F1(num_classes=40)

        self.true_pos_count = torch.zeros(40)
        self.true_neg_count = torch.zeros(40)
        self.false_pos_count = torch.zeros(40)
        self.false_neg_count = torch.zeros(40)

        # self.convolution = nn.Sequential(
        #     nn.Conv2d(3, 75, (7, 7)),
        #     nn.BatchNorm2d(75),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(75, 200, (3, 3)),
        #     nn.BatchNorm2d(200),
        #     nn.ReLU(),
        #     nn.Conv2d(200, 300, (3, 3)),
        #     nn.BatchNorm2d(300),
        #     nn.ReLU(),
        #     nn.Conv2d(300, 512, (3, 3)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 40, (3, 3)),
        #     nn.BatchNorm2d(40)
        # )
        #
        # self.fully_connected = nn.Sequential(
        #     nn.Linear(40 * 96 * 76, 40)
        # )

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 75, (7, 7)),
            nn.BatchNorm2d(75),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(75, 200, (3, 3)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 300, (3, 3)),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Conv2d(300, 512, (3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.segment = nn.Sequential(
            nn.Conv2d(512, 40, (3, 3)),
            nn.BatchNorm2d(40)
        )

        self.attributes1 = nn.Sequential(
            nn.Conv2d(512, 40, (3, 3)),
            nn.BatchNorm2d(40)
        )
        self.attributes2 = nn.Sequential(
            nn.Linear(40 * 96 * 76, 40)
        )

        # self.convolution1 = nn.Sequential(
        #     nn.Conv2d(3, 75, (7, 7)),
        #     nn.BatchNorm2d(75),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(75, 200, (3, 3)),
        #     nn.BatchNorm2d(200),
        #     nn.ReLU(),
        #     nn.Conv2d(200, 300, (3, 3)),
        #     nn.BatchNorm2d(300),
        #     nn.ReLU(),
        #     nn.Conv2d(300, 512, (3, 3)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, (3, 3)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 40, (3, 3)),
        #     nn.BatchNorm2d(40),
        # )
        #
        # self.convolution2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(40, 80, (3, 3)),
        #     nn.BatchNorm2d(80),
        # )
        #
        # self.fully_connected = nn.Sequential(
        #     nn.Linear(80 * 74 * 94, 40)
        # )

    def forward(self, x):
        # feature_maps = self.convolution(x)
        # attributes = self.fully_connected(feature_maps.view(-1, self.num_flat_features(feature_maps)))

        # feature_maps = self.convolution1(x)
        # conv2_feature_maps = self.convolution2(feature_maps)
        # attributes = self.fully_connected(conv2_feature_maps.view(-1, self.num_flat_features(conv2_feature_maps)))

        part1 = self.convolution(x)
        feature_maps = self.segment(part1)
        attributes_branch = self.attributes1(part1)
        attributes = self.attributes2(attributes_branch.view(-1, self.num_flat_features(attributes_branch)))

        return attributes, feature_maps

    def training_step(self, train_batch, batch_idx):
        if self.segment_flag == False:
            inputs, attribute_labels = train_batch['image'], train_batch['attributes']

            attribute_preds, mask_preds = self(inputs)

            mse_loss = 0
            bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')
            loss = bce_loss

        else:
            inputs, attribute_labels, mask_labels = train_batch['image'], train_batch['attributes'], train_batch['masks']

            attribute_preds, mask_preds = self(inputs)

            mse_loss = F.mse_loss(mask_preds, mask_labels, reduction='mean') * 16
            bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')
            loss = bce_loss + mse_loss

        self.log("Training Accuracy", self.train_accuracy(attribute_preds, attribute_labels), on_step=True, on_epoch=True, logger=True)
        self.log("Training Precision", self.train_precision(attribute_preds, attribute_labels), on_step=True, on_epoch=True, logger=True)
        self.log("Training Recall", self.train_recall(attribute_preds, attribute_labels), on_step=True, on_epoch=True, logger=True)
        self.log("Training F1", self.train_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=True, logger=True)

        self.log('Training Loss BCE', bce_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('Training Loss MSE', mse_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('Training Loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     self.log("Training Accuracy", self.train_accuracy.compute(), on_epoch=True, logger=True)
    #     self.log("Training Precision", self.train_precision.compute(), on_epoch=True, logger=True)
    #     self.log("Training Recall", self.train_recall.compute(), on_epoch=True, logger=True)
    #     self.log("Training F1", self.train_f1.compute(), on_epoch=True, logger=True)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):

        if self.repair_labels_flag == True:
            mask_labels = batch['masks']
            attribute_labels = batch['attributes']

            for index1, masks in enumerate(mask_labels):
                for index2, mask in enumerate(masks):
                    if attribute_labels[index1][index2] == 1:
                        num_pos_pixels = mask[mask > .5]
                        if list(num_pos_pixels.shape)[0] == 0:
                            attribute_labels[index1][index2] = 0

        if self.show_batch_flag == True:
            if self.segment_flag == True:
                batch_inputs, batch_attribute_labels, batch_mask_labels = batch['image'], batch['attributes'], batch['masks']
            else:
                batch_inputs, batch_attribute_labels = batch['image'], batch['attributes']

            ########## SHOW INPUT ##########
            for index, image in enumerate(batch_inputs):
                image = image.numpy()
                image = image.transpose(1, 2, 0)
                # Note that there are two ways to view an image. Save the image and open it, or
                #      show the image while the program is running. Either uncomment imshow and waitKey
                #      or imwrite
                plt.imshow(image)
                plt.show()
                input("Press 'Enter' for next input image.")

            if self.segment_flag == True:
                ########## SHOW MASKS ##########
                for index1, sample_masks in enumerate(batch_mask_labels):
                    for index2, mask in enumerate(sample_masks):
                        mask = (mask.numpy() * 255).astype(int)
                        # Note that there are two ways to view an image. Save the image and open it, or
                        #      show the image while the program is running. Either uncomment imshow and
                        #      waitKey or imwrite
                        plt.imshow(mask)
                        plt.show()
                    input("Press 'Enter' for next sample's masks.")

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.show_batch_flag == True:
            if self.segment_flag == True:
                batch_inputs, batch_attribute_labels, batch_mask_labels = batch['image'], batch['attributes'], batch['masks']
            else:
                batch_inputs, batch_attribute_labels = batch['image'], batch['attributes']

                ########## SHOW INPUT ##########
                for index, image in enumerate(batch_inputs):
                    image = image.numpy()
                    image = image.transpose(1, 2, 0)
                    # Note that there are two ways to view an image. Save the image and open it, or
                    #      show the image while the program is running. Either uncomment imshow and waitKey
                    #      or imwrite
                    plt.imshow(image)
                    plt.show()
                    input("Press 'Enter' for next input image.")
                    # cv2.imwrite(f"batch_image_{index}.png", image)

                if self.segment_flag == True:
                    ########## SHOW MASKS ##########
                    for index1, sample_masks in enumerate(batch_mask_labels):
                        for index2, mask in enumerate(sample_masks):
                            mask = (mask.numpy() * 255).astype(int)
                            # Note that there are two ways to view an image. Save the image and open it, or
                            #      show the image while the program is running. Either uncomment imshow and
                            #      waitKey or imwrite
                            plt.imshow(mask)
                            plt.show()

                            # mask_prediction = (outputs)

                        input("Press 'Enter' for next sample's masks.")

    def validation_step(self, val_batch, batch_idx):
        self.evaluating_flag = True
        inputs, attribute_labels = val_batch['image'], val_batch['attributes']
        attribute_preds, mask_preds = self(inputs)

        mse_loss = 0
        bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

        loss = bce_loss

        # self.compute_metric_counts(attribute_preds, attribute_labels)

        self.log("Validation Accuracy", self.valid_accuracy(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation Precision", self.valid_precision(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation Recall", self.valid_recall(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Validation F1", self.valid_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)

        self.log('Validation Loss BCE', bce_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log('Validation Loss MSE', mse_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log('Validation Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

    # def validation_epoch_end(self, outputs):
    #     self.compute_metrics()
    #     self.output(True, False)
    #     self.log("Validation Accuracy", self.valid_accuracy.compute(), on_epoch=True, logger=True)
    #     self.log("Validation Precision", self.valid_precision.compute(), on_epoch=True, logger=True)
    #     self.log("Validation Recall", self.valid_recall.compute(), on_epoch=True, logger=True)
    #     self.log("Validation F1", self.valid_f1.compute(), on_epoch=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        self.evaluating_flag = True
        inputs, attribute_labels = test_batch['image'], test_batch['attributes']
        attribute_preds, mask_preds = self(inputs)

        # self.compute_metric_counts(attribute_preds, attribute_labels)

        mse_loss = 0
        bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

        loss = bce_loss

        self.log("Test Accuracy", self.test_accuracy(torch.round(torch.sigmoid(attribute_preds)), attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Test Precision", self.test_precision(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Test Recall", self.test_recall(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("Test F1", self.test_f1(attribute_preds, attribute_labels), on_step=True, on_epoch=True, sync_dist=True, logger=True)

        self.log('Test Loss BCE', bce_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log('Test Loss MSE', mse_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.log('Test Loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        # if self.save_feature_map_flag:
        #     pos_feature_map_avg = torch.zeros((40, 76, 96))
        #     pos_count = torch.zeros(40)
        #     neg_feature_map_avg = torch.zeros((40, 76, 96))
        #     neg_count = torch.zeros(40)
        #     for index1, sample_masks in enumerate(mask_preds):
        #         if not os.path.exists("./feature_maps/attparsenet/input/"):
        #             os.mkdir("./feature_maps/attparsenet/input/")
        #         if not os.path.exists("./feature_maps/attparsenet/pos/"):
        #             os.mkdir("./feature_maps/attparsenet/pos/")
        #         if not os.path.exists("./feature_maps/attparsenet/neg/"):
        #             os.mkdir("./feature_maps/attparsenet/neg/")
        #
        #         image = inputs[index1].cpu().numpy().transpose(1, 2, 0)
        #         plt.imsave(f"./feature_maps/attparsenet/input/batch_{batch_idx}_sample_{index1}_input.png", image,
        #                    cmap="Greys")
        #         for index2, mask in enumerate(sample_masks):
        #             if attribute_labels[index1][index2] == 1:
        #                 mask = mask.cpu().numpy()
        #                 pos_feature_map_avg[index2] += mask
        #                 pos_count[index2] += 1
        #                 plt.imsave(f"./feature_maps/attparsenet/pos/batch_{batch_idx}_sample_{index1}_attribute_{self.attr_list[index2]}_mask.png", mask, cmap="Greys")
        #             else:
        #                 mask = mask.cpu().numpy()
        #                 neg_feature_map_avg[index2] += mask
        #                 neg_count[index2] += 1
        #                 plt.imsave(f"./feature_maps/attparsenet/neg/batch_{batch_idx}_sample_{index1}_attribute_{self.attr_list[index2]}_mask.png", mask, cmap="Greys")
        #
        # for index3, attr in enumerate(self.attr_list):
        #     plt.imsave(f"./feature_maps/attparsenet/pos_avg_{attr}_mask.png", (pos_feature_map_avg[index3]/pos_count[index3]).cpu().numpy(), cmap="Greys")
        #     plt.imsave(f"./feature_maps/attparsenet/neg_avg_{attr}_mask.png", (neg_feature_map_avg[index3]/pos_count[index3]).cpu().numpy(), cmap="Greys")

    # def test_epoch_end(self, outputs):
    #     self.compute_metrics()
    #     print(self.class_accuracy, self.class_accuracy_pos, self.class_accuracy_neg, self.class_recall, self.class_precision, self.class_f1)
    #     self.output(file=True)
    #     self.log("Test Accuracy", self.test_accuracy.compute(), logger=True)
    #     self.log("Test Precision", self.test_precision.compute(), logger=True)
    #     self.log("Test Recall", self.test_recall.compute(), logger=True)
    #     self.log("Test F1", self.test_f1.compute(), logger=True)

    def compute_metric_counts(self, attribute_preds, attribute_labels):
        # We use the BCEWithLogits loss function, so the sigmoid needs to be applied before computing our metrics
        attribute_preds = torch.sigmoid(attribute_preds)
        attribute_preds = torch.round(attribute_preds)

        # Remove the predictions from GPU and move to CPU
        # attribute_preds_cpu = attribute_preds.detach().to(torch.device("cpu"))
        # attribute_labels_cpu = attribute_labels.detach().to(torch.device("cpu"))
        attribute_preds_cpu = attribute_preds.cpu()
        attribute_labels_cpu = attribute_labels.cpu()

        attribute_positive_preds = torch.ge(attribute_preds_cpu, 1)
        attribute_negative_preds = torch.lt(attribute_positive_preds, 1)
        attribute_positive_labels = torch.ge(attribute_labels_cpu, 1)
        attribute_negative_labels = torch.lt(attribute_positive_labels, 1)

        true_positive = torch.sum(torch.logical_and(attribute_positive_preds, attribute_positive_labels).int(), dim=0)
        false_positive = torch.sum(torch.logical_and(attribute_positive_preds, attribute_negative_labels).int(), dim=0)
        true_negative = torch.sum(torch.logical_and(attribute_negative_preds, attribute_negative_labels).int(), dim=0)
        false_negative = torch.sum(torch.logical_and(attribute_negative_preds, attribute_positive_labels).int(), dim=0)

        self.true_pos_count += true_positive
        self.true_neg_count += true_negative
        self.false_pos_count += false_positive
        self.false_neg_count += false_negative

    # Computes total accuracy, accuracy of positive samples, accuracy of negative samples, precision, and recall of the
    # current batch
    def compute_metrics(self):
        self.class_precision = (self.true_pos_count / (self.true_pos_count + self.false_pos_count))
        self.class_recall = (self.true_pos_count / (self.true_pos_count + self.false_neg_count))
        self.class_f1 = 2 * ((self.class_precision * self.class_recall) / (self.class_precision + self.class_recall))
        self.class_accuracy = ((self.true_pos_count + self.true_neg_count) / (
                self.true_pos_count + self.true_neg_count + self.false_pos_count + self.false_neg_count))
        self.class_accuracy_pos = (self.true_pos_count / (self.true_pos_count + self.false_neg_count))
        self.class_accuracy_neg = (self.true_neg_count / (self.true_neg_count + self.false_pos_count))

        self.class_precision = torch.where(torch.isnan(self.class_precision), torch.zeros_like(self.class_precision), self.class_precision)
        self.class_recall = torch.where(torch.isnan(self.class_recall), torch.zeros_like(self.class_recall), self.class_recall)
        self.class_f1 = torch.where(torch.isnan(self.class_f1), torch.zeros_like(self.class_f1), self.class_f1)
        self.class_accuracy = torch.where(torch.isnan(self.class_accuracy), torch.zeros_like(self.class_accuracy), self.class_accuracy)
        self.class_accuracy_pos = torch.where(torch.isnan(self.class_accuracy_pos), torch.zeros_like(self.class_accuracy_pos), self.class_accuracy_pos)
        self.class_accuracy_neg = torch.where(torch.isnan(self.class_accuracy_neg), torch.zeros_like(self.class_accuracy_neg), self.class_accuracy_neg)

    # Function for outputting metrics. Can output to console, csv, or txt (note comma separated) file
    def output(self, file=False, csv=False):
        ### TO FILE ###
        if file == True:
            fout = open(
                f"Baseline_mult_hflip_epoch17.txt", "w+")
            fout.write(
                "{0:29} {1:13} {2:13} {3:13} {4:13} {5:13} {6:13}\n".format("\nAttributes", "Acc", "Acc_pos", "Acc_neg",
                                                                            "Precision", "Recall", "F1"))
            fout.write('-' * 103)
            fout.write("\n")

            for attr, acc, acc_pos, acc_neg, prec, rec, f1 in zip(self.attr_list, self.class_accuracy,
                                                                  self.class_accuracy_pos, self.class_accuracy_neg,
                                                                  self.class_precision, self.class_recall,
                                                                  self.class_f1):
                fout.write(
                    "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format(attr, acc, acc_pos,
                                                                                                  acc_neg, prec, rec,
                                                                                                  f1))
            fout.write('-' * 103)
            fout.write('\n')

            fout.write(
                "{0:19} {1:13.3f} {2:13.3f} {3:13.3f} {4:13.3f} {5:13.3f} {6:13.3f}\n".format('', torch.mean(self.class_accuracy),
                                                                                              torch.mean(self.class_accuracy_pos),
                                                                                              torch.mean(self.class_accuracy_neg),
                                                                                              torch.mean(self.class_precision),
                                                                                              torch.mean(self.class_recall),
                                                                                              torch.mean(self.class_f1)))
            fout.close()
        ##############

        ### TO CSV ###
        output_preds = []
        if csv == True:
            output_preds.append(self.class_accuracy.tolist())
            output_preds.append(self.class_accuracy_pos.tolist())
            output_preds.append(self.class_accuracy_neg.tolist())
            output_preds.append(self.class_precision.tolist())
            output_preds.append(self.class_recall.tolist())
            output_preds.append(self.class_f1_score.tolist())
            output_preds_df = pd.DataFrame(output_preds)
            output_preds_df.to_csv(f"Baseline_mult_hflip_epoch17.csv", sep=',')
        ##############

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        lmbda = lambda epoch: 0.9
        lr_scheduler = {
            # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=self.patience, verbose=False),
            # 'monitor': 'Validation Loss'
            'scheduler': optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False),
        }

        return [optimizer], [lr_scheduler]

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features