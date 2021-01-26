import torch
import wandb
import attparsenet_utils
import attparsenet
import attparsenet_dataset
import attparsenet_random_crop

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from pytorch_lightning.loggers import WandbLogger


from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

# wandb.init(project="attparsenet", entity="unr-mpl")

# Base Model, Dataset, Batch Size, Learning Rate
wandb_logger = WandbLogger(name='AttParseNet Unaligned 40 0.01 Mworks08', project='attparsenet', entity='unr-mpl')

activation = None

if __name__=="__main__":
    args = attparsenet_utils.get_args()

    pl.seed_everything(2)

    # Initialize the model
    if args.model == "attparsenet":
        net = attparsenet.AttParseNet(args)

        training_dataset = attparsenet_dataset.AttParseNetDataset(
            args.segment, False, args.image_path, args.image_dir, args.mask_image_path, args.attr_label_path,
            args.mask_label_path, transform=transforms.Compose(
                [attparsenet_random_crop.AttParseNetRandomCrop((178, 218), (76, 96), args.segment, False)]
            ))

        evaluating_dataset = attparsenet_dataset.AttParseNetDataset(
            args.segment, True, args.image_path, args.image_dir, args.mask_image_path, args.attr_label_path,
            args.mask_label_path, transform=transforms.Compose(
                [attparsenet_random_crop.AttParseNetRandomCrop((178, 218), (76, 96), args.segment, True)]
            ))

    if args.shuffle:
        train_indices, val_indices, test_indices = (torch.randperm(args.train_size),
                                                    torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                    torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))
    else:
        train_indices, val_indices, test_indices = (torch.tensor(list(range(args.train_size))),
                                                   torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                   torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))

    # train_indices = list(range(35000))
    # val_indices = list(range(10000))

    train_set = Subset(training_dataset, train_indices)
    val_set = Subset(evaluating_dataset, val_indices)
    test_set = Subset(evaluating_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='Validation Loss_epoch',
        dirpath='/home/nthom/Documents/AttParseNet/checkpoints',
        filename='AttParseNet_Unaligned_40_0.01_Mworks08-{epoch:02d}-{Validation Loss_epoch:.05f}',
        save_top_k=50,
        mode='min',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='ddp',
        precision=16,
        # checkpoint_callback=False,
        # checkpoint_callback=True,
        callbacks=[checkpoint_callback],
 #        val_check_interval=0.25,
        limit_train_batches=1.0,
        # limit_val_batches=0.05,
        gpus=2,
        max_epochs=args.train_epochs
    )

    # trainer.tune(net)

    trainer.fit(net, train_loader, val_loader)
