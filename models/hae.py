from .mish import Mish
from .r_adam import RAdam
from .utils import device
from .scheduler import FlatCA

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import GELU



# ARCHITECTURE

class Encoder(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim, hidden_dim // 2,
                      kernel_size=3, stride=2, padding=1),
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
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim  # num channels on bottom layer

        # Mish()]
        blocks = [nn.Conv2d(in_feat_dim, hidden_dim,
                            kernel_size=3, padding=1), GELU()]

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
        self.mish = Mish()
    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.mish(x)
        x = self.conv_2(x)
        x = x + inp
        return self.mish(x)


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
        self.register_buffer(
            "running_ave", torch.zeros(1, self.feature_dim, 1, 1))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer(
                "running_sq_diff", torch.zeros(1, self.feature_dim, 1, 1))

    def forward(self, inputs):

        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * \
                inputs.shape[2] * inputs.shape[3]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen +
                inputs.sum(dim=(0, 2, 3), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)

            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) *
                    (inputs - updated_running_ave)
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


class HAE(pl.LightningModule):
    def __init__(
        self,
        prev_model,
        input_feat_dim,
        lr=4e-4,
        codebook_slots=256,
        codebook_dim=64,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        gs_temp=0.667,
        num_res_blocks=0
    ):
        super().__init__()
        self.lr = lr
        self.prev_model = prev_model
        self.encoder = Encoder(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks)
        self.decoder = Decoder(
            codebook_dim,
            input_feat_dim,
            dec_hidden_dim,
            num_res_blocks=num_res_blocks,
            very_bottom=prev_model is None,
        )
        self.codebook_dim = codebook_dim
        self.normalize = GlobalNormalization(codebook_dim, scale=True)
        self.save_hyperparameters(ignore=['prev_model'])

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder,
                       # self.codebook,
                       self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param

    @classmethod
    def init_higher(cls, prev_model, **kwargs) -> pl.LightningModule:
        model = HAE(prev_model,
                    prev_model.codebook_dim,
                    **kwargs)
        model.prev_model.eval()
        return model

    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HAE(None, input_feat_dim, **kwargs)
        return model

    def forward(self, img):
        z_e_lower = self.encode_lower(img)
        z_e = self.encoder(z_e_lower)
        # import ipdb; ipdb.set_trace()
        # z_e_lower_tilde = self.decoder(z_q)
        z_e_lower_tilde = self.decoder(z_e)
        return z_e_lower_tilde, z_e_lower, z_e

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, orig, _ = self(x)
        recon_loss = self.recon_loss(orig, recon)

        # calculate loss
        dims = np.prod(recon.shape[1:])  # orig_w * orig_h * num_channels
        loss = recon_loss/dims

        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, orig, _ = self(x)
        recon_loss = self.recon_loss(orig, recon)

        # calculate loss
        dims = np.prod(recon.shape[1:])  # orig_w * orig_h * num_channels
        loss = recon_loss/dims

        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.lr)
        scheduler = FlatCA(
            optimizer, steps=self.trainer.estimated_stepping_batches, eta_min=self.lr/10)
        return [optimizer], [scheduler]

    def forward_full_stack(self, img):
        z_e = self.encode(img)
        img_recon_dist = self.decode(z_e)
        return img_recon_dist, img, z_e

    def encode_lower(self, x):
        if self.prev_model is None:
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
                recon = self.decode_lower(z_e_u)
            else:
                recon = self.decoder(z_q)
        return recon

    def reconstruct_average(self, x, num_samples=10):
        """Average over stochastic edecodes"""
        b, c, h, w = x.shape
        result = torch.empty((num_samples, b, c, h, w)).to(device)

        for i in range(num_samples):
            result[i] = self.decode(self.encode(x))
        return result.mean(0)

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def reconstruct_from_z_e(self, z_e):
        return self.decode(z_e)

    def recon_loss(self, orig, recon):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1, 2, 3)).mean()

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
