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

# wandb.init(project="attparsenet", entity="unr-mpl")
wandb_logger = WandbLogger(name='AttParseNet Unaligned 64 0.1', project='attparsenet', entity='unr-mpl')

activation = None

if __name__=="__main__":
    args = attparsenet_utils.get_args()

    pl.seed_everything(64)

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
        train_indices, val_indices, test_indices = (torch.randint(low=0, high=args.train_size, size=(args.train_size,)),
                                                    torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                    torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))
    else:
        train_indices, val_indices, test_indices = (torch.tensor(list(range(args.train_size))),
                                                   torch.tensor(list(range(args.train_size, args.train_size + args.val_size))),
                                                   torch.tensor(list(range(args.train_size + args.val_size, args.all_size))))

    train_indices = list(range(5000))
    val_indices = list(range(1000))

    train_set = Subset(training_dataset, train_indices)
    val_set = Subset(evaluating_dataset, val_indices)
    test_set = Subset(evaluating_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='ddp',
        precision=16,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        checkpoint_callback=True,
        gpus=2,
        max_epochs=args.train_epochs
    )

    # trainer.tune(net)

    trainer.fit(net, train_loader, val_loader)