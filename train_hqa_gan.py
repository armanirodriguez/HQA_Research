# Armani Rodriguez

from Datasets.ImageNetDataset import ImageNet100
from armani.DL_class.models.hqa_gan.hqa_gan import HQAGAN

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

import os


DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'


def load_hqa_from_checkpoints(
    n_layers=5,
    checkpoints_dir='hqa_gan_checkpoints'
) -> pl.LightningModule:
    """
    Load a HAE model from a directory of checkpoints for each layer
    """
    for layer in range(n_layers):
        if layer == 0:
            hqa = HQAGAN.load_from_checkpoint(
                f"{checkpoints_dir}/hqa_gan_layer{layer}.ckpt", prev_model=None)
        else:
            hqa = HQAGAN.load_from_checkpoint(
                f"{checkpoints_dir}/hqa_gan_layer{layer}.ckpt", prev_model=hqa_prev)
        hqa_prev = hqa
    return hqa


def get_dataloader(training=True) -> DataLoader:
    if training:
        dataset = ImageNet100(DATA_FOLDER, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
    else:
        dataset = ImageNet100(DATA_FOLDER, split="val", transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))

    dataloader = DataLoader(dataset, batch_size=64, num_workers=8)
    return dataloader


def train_full_stack(training_dataloader, 
                     validation_dataloader,
                     input_feat_dim,
                     epochs,
                     n_layers,
                     enc_hidden_sizes=[16, 16, 32, 64, 128], 
                     dec_hidden_sizes=[16, 64, 256, 512, 1024],
                     dev_run=False,
                     num_res_blocks=0):
    n_layers = min(len(enc_hidden_sizes), n_layers)
    for layer in range(n_layers):
        if layer == 0:
            hqa = HQAGAN.init_bottom(input_feat_dim=input_feat_dim,
                                  enc_hidden_dim=enc_hidden_sizes[layer],
                                  dec_hidden_dim=dec_hidden_sizes[layer],
                                  num_res_blocks=num_res_blocks)
        else:
            hqa = HQAGAN.init_higher(hqa_prev,
                                  enc_hidden_dim=enc_hidden_sizes[layer],
                                  dec_hidden_dim=dec_hidden_sizes[layer])

        print(f"Training layer {layer}:")

        if layer == 0:
            trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_run, devices=1)
        else:
            trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_run,devices=1)

        trainer.fit(hqa, train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)
        trainer.save_checkpoint(f"hqa_gan_checkpoints/hqa_gan_layer{layer}.ckpt")
        hqa_prev = hqa
    return hqa


def train(epochs=20, dev_run=False, checkpoint_dir='hqa_gan_checkpoints', n_layers=5) -> pl.LightningModule:
    """
    Train hae model on ImageNet100
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.set_float32_matmul_precision('medium')

    training_dataloader = get_dataloader()
    validation_dataloader = get_dataloader(training=False)

    hqa = train_full_stack(training_dataloader, validation_dataloader, input_feat_dim=3, epochs=epochs, n_layers=n_layers, dev_run=dev_run)
    return hqa


if __name__ == '__main__':
    train(epochs=20, n_layers=1) 
