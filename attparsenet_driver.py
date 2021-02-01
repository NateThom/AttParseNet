import torch
import wandb
import attparsenet_utils
import attparsenet
import attparsenet_dataset
import attparsenet_random_crop
import attparsenet_random_horizontal_flip

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

# Base Model, Dataset, Batch Size, Learning Rate
wandb_logger = WandbLogger(name='AttParseNet Unaligned Mult hflip 40 0.01 Mworks08', project='attparsenet', entity='unr-mpl')

activation = None

if __name__=="__main__":
    args = attparsenet_utils.get_args()

    pl.seed_everything(args.random_seed)

    # Initialize the model
    if args.model == "attparsenet":

        if args.load == True:
            net = attparsenet.AttParseNet.load_from_checkpoint(args.load_path + args.load_file)
        else:
            net = attparsenet.AttParseNet(args)

        training_dataset = attparsenet_dataset.AttParseNetDataset(
            args.segment, False, args.image_path, args.image_dir, args.mask_image_path, args.attr_label_path,
            args.mask_label_path, transform=transforms.Compose(
                [attparsenet_random_crop.AttParseNetRandomCrop((178, 218), (76, 96), args.segment, False),
                 attparsenet_random_horizontal_flip.AttParseNetHorizontalFlip(args.segment, False)]
            ))

        evaluating_dataset = attparsenet_dataset.AttParseNetDataset(
            args.segment, True, args.image_path, args.image_dir, args.mask_image_path, args.attr_label_path,
            args.mask_label_path, transform=transforms.Compose(
                [attparsenet_random_crop.AttParseNetRandomCrop((178, 218), (76, 96), args.segment, True),
                 attparsenet_random_horizontal_flip.AttParseNetHorizontalFlip(args.segment, True)]
            ))

    if args.shuffle:
        train_indices, val_indices, test_indices = (torch.randperm(args.train_size),
                                                    torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                    torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))
    else:
        train_indices, val_indices, test_indices = (torch.tensor(list(range(args.train_size))),
                                                   torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                   torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))

    train_set = Subset(training_dataset, train_indices)
    val_set = Subset(evaluating_dataset, val_indices)
    test_set = Subset(evaluating_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Validation Loss',
            dirpath=args.save_path,
            filename='AttParseNet_Unaligned_mult_hflip_40_0.01_Mworks08-{epoch:02d}-{Validation Loss:.05f}',
            save_top_k=50,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            gpus=1,
            max_epochs=args.train_epochs
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            checkpoint_callback=False,
            # accelerator='ddp',
            gpus=1,
            # num_nodes=1,
            # limit_train_batches=0.1,
            max_epochs=args.train_epochs
        )

    if args.train == True:
        trainer.fit(net, train_loader, val_loader)
        # trainer.fit(net, train_loader)

    if args.val_only == True:
        trainer.test(net, val_loader)

    if args.test == True:
        trainer.test(net, test_loader)