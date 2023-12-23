import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import RelaxedOneHotCategorical, Normal, Categorical
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from torchvision import transforms
from torchvision.datasets import MNIST

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

import albumentations as A 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)
    
class FlatCA(_LRScheduler):
    def __init__(self, optimizer, steps, eta_min=0, last_epoch=-1):
        self.steps = steps
        self.eta_min = eta_min
        super(FlatCA, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        T_max = self.steps / 3
        for base_lr in self.base_lrs:
            # flat if first 2/3
            if 0 <= self._step_count < 2 * T_max:
                lr_list.append(base_lr)
            # annealed if last 1/3
            else:
                lr_list.append(
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * (self._step_count - 2 * T_max) / T_max))
                    / 2
                )
            return lr_list

class Encoder(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=3):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.append(nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """ Upsamples by a fac of 2 """

    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=3, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer

        blocks = [nn.Conv2d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv2d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
        ])

        if very_bottom is True:
            blocks.append(nn.Sigmoid())       
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = mish(x)
        x = self.conv_2(x)
        x = x + inp
        return mish(x)

class VQCodebook(nn.Module):
    def __init__(self, codebook_slots, codebook_dim, temperature=0.5):
        super().__init__()
        self.codebook_slots = codebook_slots
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.codebook = nn.Parameter(torch.randn(codebook_slots, codebook_dim))
        self.log_slots_const = np.log(self.codebook_slots)

    def z_e_to_z_q(self, z_e, soft=True):
        bs, feat_dim, w, h = z_e.shape
        assert feat_dim == self.codebook_dim
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e.view(bs * w * h, feat_dim)
        codebook_sqr = torch.sum(self.codebook ** 2, dim=1)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)

        distances = torch.addmm(
            codebook_sqr + z_e_flat_sqr, z_e_flat, self.codebook.t(), alpha=-2.0, beta=1.0
        )

        if soft is True:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, w, h)
            z_q = (soft_onehot @ self.codebook).view(bs, w, h, feat_dim)
            
            # entropy loss
            KL = dist.probs * (dist.probs.add(1e-9).log() + self.log_slots_const)
            KL = KL.view(bs, w, h, self.codebook_slots).sum(dim=(1,2,3)).mean()
            
            # probability-weighted commitment loss    
            commit_loss = (dist.probs.view(bs, w, h, self.codebook_slots) * distances.view(bs, w, h, self.codebook_slots)).sum(dim=(1,2,3)).mean()
        else:
            with torch.no_grad():
                dist = Categorical(logits=-distances)
                hard_indices = dist.sample().view(bs, w, h)
                hard_onehot = (
                    F.one_hot(hard_indices, num_classes=self.codebook_slots)
                    .type_as(self.codebook)
                    .view(bs * w * h, self.codebook_slots)
                )
                z_q = (hard_onehot @ self.codebook).view(bs, w, h, feat_dim)
                
                # entropy loss
                KL = dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))
                KL = KL.view(bs, w, h, self.codebook_slots).sum(dim=(1,2,3)).mean()

                commit_loss = 0.0

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, hard_indices, KL, commit_loss

    def lookup(self, ids: torch.Tensor):
        return F.embedding(ids, self.codebook).permute(0, 3, 1, 2)

    def quantize(self, z_e, soft=False):
        with torch.no_grad():
            z_q, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return z_q, indices

    def quantize_indices(self, z_e, soft=False):
        with torch.no_grad():
            _, indices, _, _ = self.z_e_to_z_q(z_e, soft=soft)
        return indices

    def forward(self, z_e):
        z_q, indices, kl, commit_loss = self.z_e_to_z_q(z_e, soft=True)
        return z_q, indices, kl, commit_loss

class GlobalNormalization(torch.nn.Module):
    """
    nn.Module to track and normalize input variables, calculates running estimates of data
    statistics during training time.
    Optional scale parameter to fix standard deviation of inputs to 1
    Normalization atlassian page:
    https://speechmatics.atlassian.net/wiki/spaces/INB/pages/905314814/Normalization+Module
    Implementation details:
    "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    """

    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, self.feature_dim, 1, 1))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer("running_sq_diff", torch.zeros(1, self.feature_dim, 1, 1))

    def forward(self, inputs):
        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen + inputs.sum(dim=(0, 2, 3), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)

            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) * (inputs - updated_running_ave)
                ).sum(dim=(0, 2, 3), keepdim=True)

            self.running_ave = updated_running_ave
            self.total_frames_seen = self.total_frames_seen + frames_in_input

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs

    def unnorm(self, inputs):
        
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs*std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs

class HQA(pl.LightningModule):
    def __init__(
        self,
        input_feat_dim,
        prev_model=None,
        codebook_slots=256,
        codebook_dim=128,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        gs_temp=0.9, #0.667,
        signal_input=False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])
        self.prev_model = prev_model
        self.encoder = Encoder(input_feat_dim, codebook_dim, enc_hidden_dim)
        self.codebook = VQCodebook(codebook_slots, codebook_dim, gs_temp)
        self.decoder = Decoder(
            codebook_dim,
            input_feat_dim,
            dec_hidden_dim,
            very_bottom=prev_model is None,
        )
        self.normalize = GlobalNormalization(codebook_dim, scale=True)
        self.signal_input = signal_input
        self.out_feat_dim = input_feat_dim

    def forward(self, x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, _, _ = self.codebook.z_e_to_z_q(z_e, soft=False)
        z_e_lower_tilde = self.decoder(z_q)
        return z_e_lower_tilde, indices
    
    def training_step(self,batch, batch_idx):
        x,_ = batch
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        recon_loss = F.mse_loss(z_e_lower, z_e_lower_tilde, reduction='none').sum(dim=(1,2,3)).mean()
        dims = np.prod(z_e_lower_tilde.shape[1:])
        loss = recon_loss/dims + 0.001*kl/dims + 0.001*(commit_loss)/dims
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        x,_ = batch
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        recon_loss = F.mse_loss(z_e_lower, z_e_lower_tilde, reduction='none').sum(dim=(1,2,3)).mean()
        dims = np.prod(z_e_lower_tilde.shape[1:])
        loss = recon_loss/dims + 0.001*kl/dims + 0.001*(commit_loss)/dims
        self.log("validation_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        lr_scheduler = FlatCA(optimizer, steps=1, eta_min=4e-5)
        return [optimizer], [lr_scheduler]
    
    def encode_lower(self, x):
        if self.prev_model is None:
            if self.signal_input:
                x = torch.reshape(x, (-1, 2, 32, 32))
            return x
        else:
            with torch.no_grad():
                z_e_lower = self.prev_model.encode(x)
                z_e_lower = self.normalize(z_e_lower)
            return z_e_lower
    def encode(self, x):
        with torch.no_grad():
            z_e_lower = self.encode_lower(x)
            z_e = self.encoder(z_e_lower)
        return z_e
        
    def decode_lower(self, z_q_lower):
        with torch.no_grad():
            recon = self.prev_model.decode(z_q_lower)           
        return recon

    def decode(self, z_q):
        with torch.no_grad():
            if self.prev_model is not None:
                z_e_u = self.normalize.unnorm(self.decoder(z_q))
                z_q_lower_tilde = self.prev_model.quantize(z_e_u)
                recon = self.decode_lower(z_q_lower_tilde)
            else:
                recon = self.decoder(z_q)
                if self.signal_input:
                    recon = torch.reshape(recon, (-1, self.out_feat_dim, 1024))

        return recon

    def quantize(self, z_e):
        z_q, _ = self.codebook.quantize(z_e)
        return z_q
    
    def reconstruct_average(self, x, num_samples=10):
        """Average over stochastic edecodes"""
        b, c, h, w = x.shape
        result = torch.empty((num_samples, b, c, h, w)).to(device)

        for i in range(num_samples):
            result[i] = self.decode(self.quantize(self.encode(x)))
        return result.mean(0)

    def reconstruct(self, x):
        return self.decode(self.quantize(self.encode(x)))
    
    def reconstruct_from_codes(self, codes):
        return self.decode(self.codebook.lookup(codes))
    
    def reconstruct_from_z_e(self, z_e):
        return self.decode(self.quantize(z_e))
    
    def __len__(self):
        i = 1
        layer = self
        while layer.prev_model is not None:
            i += 1
            layer = layer.prev_model
        return i

    def __getitem__(self, idx):
        max_layer = len(self) - 1
        if idx > max_layer:
            raise IndexError("layer does not exist")

        layer = self
        for _ in range(max_layer - idx):
            layer = layer.prev_model
        return layer

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.codebook, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param
    
    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HQA(prev_model.codebook.codebook_dim, prev_model=prev_model, **kwargs)
        return model
    
    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HQA(input_feat_dim, prev_model=None, **kwargs)
        return model

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),

            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ]
    )
    from ImageNetDataset import ImageNet100
    batch_size = 128
    #ds_train = MNIST('./data/mnist', download=True, train=True, transform=train_transform)
    #dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    #ds_test = MNIST('./data/mnist', download=True, train=False, transform=val_transform)
    #dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
    DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'
    dataset_train = ImageNet100(DATA_FOLDER, split="train", remap_labels=False, transform=transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),]))
    dataset_test = ImageNet100(DATA_FOLDER, split="val", remap_labels=False, transform=transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]))

    dl_train = DataLoader(dataset_train, batch_size=64, num_workers=8)
    dl_test = DataLoader(dataset_test, batch_size=64, num_workers=8)

    enc_hidden_sizes = [64, 64, 128, 256, 512]
    dec_hidden_sizes = [64, 256, 512, 1024, 2048]

    EPOCHS=20
    LAYERS = 5
    FAST_DEV_RUN = False
    CHANNELS=3

    for i in range(LAYERS):
        print("Training layer", i)
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=CHANNELS,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_dim=512
            )
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_dim=512
            )
        logger = TensorBoardLogger("tb_logs", name="HQA_Imagnet")
        if i > 0:
            trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, devices=2, num_sanity_val_steps=0, strategy='ddp_find_unused_parameters_true')
        else:
            trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, devices=2, num_sanity_val_steps=0)
        trainer.fit(model=hqa, train_dataloaders=dl_train, val_dataloaders=dl_test)
        if not FAST_DEV_RUN:
            trainer.save_checkpoint(f'./hqa_imagenet_checkpoints/layer{i}.ckpt')
        hqa_prev = hqa
