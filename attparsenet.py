import torch
import wandb
import time
import matplotlib

from torch import nn

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class AttParseNet(pl.LightningModule):
    def __init__(self, args):
        super(AttParseNet, self).__init__()

        assert isinstance(args.segment, (bool))
        self.segment_flag = args.segment
        assert isinstance(args.evaluating, (bool))
        self.evaluating_flag = args.evaluating
        assert isinstance(args.show_batch, (bool))
        self.show_batch_flag = args.show_batch
        assert isinstance(args.repair_labels, (bool))
        self.repair_labels_flag = args.repair_labels

        assert isinstance(args.lr, (float))
        self.lr = args.lr
        assert isinstance(args.patience, (int))
        self.patience = args.patience

        # self.learning_rate = learning_rate

        self.train_accuracy = pl.metrics.Accuracy()
        self.train_precision = pl.metrics.Precision(num_classes=40)
        self.train_recall = pl.metrics.Recall(num_classes=40)
        self.train_f1 = pl.metrics.F1(num_classes=40)
        self.valid_accuracy = pl.metrics.Accuracy()
        self.valid_precision = pl.metrics.Precision(num_classes=40)
        self.valid_recall = pl.metrics.Recall(num_classes=40)
        self.valid_f1 = pl.metrics.F1(num_classes=40)

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
            nn.Conv2d(512, 40, (3, 3)),
            nn.BatchNorm2d(40)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(40 * 96 * 76, 40)
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        self.feature_maps = self.convolution(x)
        self.attributes = self.fully_connected(self.feature_maps.view(-1, self.num_flat_features(self.feature_maps)))
        return self.attributes, self.feature_maps

    def training_step(self, train_batch, batch_idx):
        if self.segment_flag == False:
            inputs, attribute_labels = train_batch['image'], train_batch['attributes']

            attribute_preds, mask_preds = self.forward(inputs)

            mse_loss = 0
            bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')
            loss = bce_loss

        else:
            inputs, attribute_labels, mask_labels = train_batch['image'], train_batch['attributes'], train_batch['masks']

            attribute_preds, mask_preds = self.forward(inputs)

            mse_loss = F.mse_loss(mask_preds, mask_labels, reduction='mean') * 8
            bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')
            loss = bce_loss + mse_loss

        self.train_accuracy(attribute_preds, attribute_labels)
        self.train_precision(attribute_preds, attribute_labels)
        self.train_recall(attribute_preds, attribute_labels)
        self.train_f1(attribute_preds, attribute_labels)

        self.log('Training Loss BCE', bce_loss, on_step=True, on_epoch=True)
        self.log('Training Loss MSE', mse_loss, on_step=True, on_epoch=True)
        self.log('Training Loss', loss, on_step=True, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        self.log("Training Accuracy", self.train_accuracy.compute(), on_epoch=True)
        self.log("Training Precision", self.train_precision.compute(), on_epoch=True)
        self.log("Training Recall", self.train_recall.compute(), on_epoch=True)
        self.log("Training F1", self.train_f1.compute(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=self.patience, verbose=False),
            'monitor': 'Validation Loss'
        }
        return [optimizer], [lr_scheduler]

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

            input("Here")

    def validation_step(self, val_batch, batch_idx):
        self.evaluating_flag = True
        inputs, attribute_labels = val_batch['image'], val_batch['attributes']
        attribute_preds, mask_preds = self.forward(inputs)

        mse_loss = 0
        bce_loss = F.binary_cross_entropy_with_logits(attribute_preds, attribute_labels, reduction='mean')

        loss = bce_loss

        self.valid_accuracy(attribute_preds, attribute_labels)
        self.valid_precision(attribute_preds, attribute_labels)
        self.valid_recall(attribute_preds, attribute_labels)
        self.valid_f1(attribute_preds, attribute_labels)

        self.log('Validation Loss BCE', bce_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('Validation Loss MSE', mse_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('Validation Loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        # return loss

    def validation_epoch_end(self, outputs):
        self.log("Validation Accuracy", self.valid_accuracy.compute(), on_epoch=True)
        self.log("Validation Precision", self.valid_precision.compute(), on_epoch=True)
        self.log("Validation Recall", self.valid_recall.compute(), on_epoch=True)
        self.log("Validation F1", self.valid_f1.compute(), on_epoch=True)

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

                            mask_prediction = (outputs)

                        input("Press 'Enter' for next sample's masks.")
