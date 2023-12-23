import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import RelaxedOneHotCategorical, Categorical
from torch.optim.lr_scheduler import _LRScheduler

from torchvision import transforms
from torchvision.datasets import MNIST

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from .lpips import LPIPS

"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, image_channels=1, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(x)

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
                    * (1 + np.cos(np.pi * (self._step_count - 2 * T_max) / T_max))
                    / 2
                )
            return lr_list

class Encoder(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0):
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
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
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
            blocks.append(Swish())
        
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

class HQAGAN(pl.LightningModule):
    def __init__(
        self,
        image_channels,
        prev_model=None,
        codebook_slots=256,
        codebook_dim=64,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        gs_temp=0.667,
        num_res_blocks=0,
        decay=True,
        hqa_lr=4e-4,                # hqa learning rate
        discriminator_lr=1e-5,      # discriminator learning rate
        discriminator_factor=0.05,   # coefficient for the discriminator loss in the loss term
        discriminator_start=10000,      # number of steps before discriminator is activated
        use_lpips=True,
        use_lambda=True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])

        # architecture
        self.prev_model = prev_model

        if prev_model is None:
            input_feat_dim = image_channels
        else:
            input_feat_dim = prev_model.codebook.codebook_dim
        
        self.encoder = Encoder(
            input_feat_dim, 
            codebook_dim, 
            enc_hidden_dim, 
            num_res_blocks=num_res_blocks
        )
        self.codebook = VQCodebook(codebook_slots, codebook_dim, gs_temp)
        self.decoder = Decoder(
            codebook_dim,
            input_feat_dim,
            dec_hidden_dim,
            very_bottom=prev_model is None,
            num_res_blocks=num_res_blocks
        )
        self.discriminator = Discriminator(image_channels=image_channels)
        #self.normalize = GlobalNormalization(codebook_dim, scale=True)

        # hyperparameters 
        self.image_channels = image_channels
        self.hqa_lr = hqa_lr
        self.discriminator_lr = discriminator_lr
        self.decay = decay
        self.discriminator_factor = discriminator_factor
        self.discriminator_start = discriminator_start
        self.use_lambda = use_lambda and (prev_model is None) # Heirarchy is interfering with the gradiant calculations, so we don't use lambda
        self.normalize = GlobalNormalization(input_feat_dim, scale=True)

        # LPIPS Perceptual Loss
        self.perceptual_loss = LPIPS().to(self.device)

        # Tells pytorch lightinig to use our custom training loop
        self.automatic_optimization = False

    
    def forward(self, x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        return z_e_lower_tilde, z_e_lower, indices, kl, commit_loss
    
    def hqa_loss(self, x, recon, KL, commit_loss):   
        recon_loss = self.recon_loss(x, recon)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = recon_loss/dims + 0.001*KL/dims + 0.001*(commit_loss)/dims
        return loss
    
    def gan_loss(self, x, recon):
        discriminator_factor = self.adopt_weight(self.discriminator_factor, self.global_step, threshold=self.discriminator_start)
        discriminator_real = self.discriminator(x)
        discriminator_fake = self.discriminator(recon)
        d_loss_real = torch.mean(F.relu(1. - discriminator_real))
        d_loss_fake = torch.mean(F.relu(1. + discriminator_fake))
        discriminator_loss = discriminator_factor * 0.5 * (d_loss_real + d_loss_fake)

        perceptual_loss = self.perceptual_loss.forward(x, recon)
        recon_loss = self.recon_loss(x, recon)
        perceptual_recon_loss = (perceptual_loss * recon_loss).mean()
        g_loss = -torch.mean(discriminator_fake)
        
        lambda_ = self.calculate_lambda(perceptual_recon_loss, g_loss) if self.use_lambda else 1

        return discriminator_loss, perceptual_recon_loss, lambda_, g_loss, discriminator_factor
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.blocks[-2 if self.decoder.very_bottom else -1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True, allow_unused=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True, allow_unused=True)[0]
        

        lambda_ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()
        return 0.8 * lambda_
        


    def recon_loss(self, orig, recon,):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1,2,3)).mean()
    
    
    def on_train_start(self):
        self.code_count = torch.zeros(self.codebook.codebook_slots, device=self.device, dtype=torch.float64)
        self.discriminator.apply(weights_init)
    
    def decay_temp_linear(self, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step/total_steps)
        return temp_min + (temp_base - temp_min) * factor

    @torch.no_grad()
    def reset_least_used_codeword(self):
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        frac_usage = self.code_count / max_count
        z_q_most_used = self.codebook.lookup(most_used_code.view(1, 1, 1)).squeeze()
        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        if min_frac_usage < 0.03:
            print(f'reset code {min_used_code}')
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / 100
            self.codebook.codebook[min_used_code] = moved_code
        self.code_count = torch.zeros_like(self.code_count, device=self.device)

    def training_step(self,batch, batch_idx):
        optimizer_hqa, optimizer_disc = self.optimizers()
        scheduler_hqa = self.lr_schedulers()

        # anneal temperature
        """
        if self.decay:
                self.codebook.temperature = self.decay_temp_linear(step = self.global_step+1, 
                                                                   total_steps = self.trainer.max_epochs * self.trainer.num_training_batches, 
                                                                    temp_base= self.codebook.temperature) 
                                                                    """
        x,_ = batch
        recon, orig, indices, kl, commit_loss = self(x)
        codebook_loss = self.hqa_loss(orig, recon, kl, commit_loss)
        
        if self.prev_model is not None:
            self.prev_model.train()
            z_q_lower_tilde, _, _, _ = self.prev_model.codebook(recon)
            x_recon = self.prev_model.decode(z_q_lower_tilde)
            self.prev_model.eval()
            discriminator_loss, perceptual_loss, lambda_, g_loss, discriminator_factor = self.gan_loss(x, x_recon)
        else:
            discriminator_loss, perceptual_loss, lambda_, g_loss, discriminator_factor = self.gan_loss(orig, recon)
        hqagan_loss = codebook_loss + discriminator_factor * lambda_ * g_loss


        optimizer_hqa.zero_grad()
        self.manual_backward(hqagan_loss, retain_graph=True)
        nn.utils.clip_grad_norm_(self.hqa_parameters(), 1.0)

        optimizer_disc.zero_grad()
        self.manual_backward(discriminator_loss)

        optimizer_disc.step()
        optimizer_hqa.step()
        scheduler_hqa.step()

        #indices_onehot = F.one_hot(indices, num_classes=self.codebook.num_codewords).float()
        #self.code_count = self.code_count + indices_onehot.sum(dim=(0, 1, 2))

        #if batch_idx > 0 and batch_idx % 20 == 0:
        #    self.reset_least_used_codeword()
        self.log("hqa_loss", hqagan_loss, prog_bar=True)
        self.log("commit_loss", commit_loss, prog_bar=True)
        self.log("kl_loss", kl, prog_bar=True)
        self.log("recon_loss", self.recon_loss(orig, recon), prog_bar=True)
        #self.log("perceptual_recon", perceptual_recon_loss, prog_bar=True)
        self.log("d_loss",discriminator_loss, prog_bar=True)
        self.log("g_loss",g_loss, prog_bar=True)

        return codebook_loss
    
    def validation_step(self, batch, batch_ndx):
        x,_ = batch
        y = self.reconstruct(x)
        loss = self.recon_loss(y,x)
        #self.log("validation_recon_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        hqa_optimizer = torch.optim.Adam(self.hqa_parameters(), lr=self.hqa_lr)
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, eps=1e-08, betas=(0.5,0.9))
        lr_scheduler = FlatCA(hqa_optimizer, steps=self.trainer.max_epochs * self.trainer.num_training_batches, eta_min=4e-5)
        return [hqa_optimizer, disc_optimizer], [lr_scheduler]

    @torch.no_grad()
    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            z_e_lower = self.prev_model.encode(x)
            #z_e_lower = self.normalize(z_e_lower)
            return z_e_lower
    
    @torch.no_grad()
    def encode(self, x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        return z_e
        

    def decode_lower(self, z_q_lower):
        #z_q_lower = self.normalize.unnorm(z_q_lower)
        recon = self.prev_model.decode(z_q_lower)           
        return recon

    def decode(self, z_q):
        if self.prev_model is not None:
            #z_e_u = self.normalize.unnorm(self.decoder(z_q))
            z_e_u = self.decoder(z_q)
            z_q_lower_tilde = self.prev_model.quantize(z_e_u)
            recon = self.decode_lower(z_q_lower_tilde)
        else:
            recon = self.decoder(z_q)
        return recon

    def quantize(self, z_e):
        z_q, _ = self.codebook.quantize(z_e)
        return z_q
    
    def reconstruct_average(self, x, num_samples=10):
        """Average over stochastic edecodes"""
        b, c, h, w = x.shape
        result = torch.empty((num_samples, b, c, h, w)).to(self.device)

        for i in range(num_samples):
            result[i] = self.decode(self.quantize(self.encode(x)))
        return result.mean(0)

    def reconstruct(self, x):
        z_e = self.encode(x)
        z_q = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon
    
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

    def hqa_parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.codebook, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
    
    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HQAGAN(prev_model.image_channels, prev_model=prev_model, **kwargs)
        model.prev_model.eval()
        return model
    
    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HQAGAN(input_feat_dim, prev_model=None, **kwargs)
        return model
    



if __name__ == '__main__':
    torch.cuda.empty_cache() 
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )
    from ImageNetDataset import ImageNet100
    batch_size = 32
    #ds_train = MNIST('./data/mnist', download=True, train=True, transform=train_transform)
    #dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    #ds_test = MNIST('./data/mnist', download=True, train=False, transform=val_transform)
    #dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
    DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'
    dataset_train = ImageNet100(DATA_FOLDER, split="train", remap_labels=False, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]))
    dataset_test = ImageNet100(DATA_FOLDER, split="val", remap_labels=False, transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))

    dl_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=8)
    dl_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=8)

    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    LAYERS = 5
    FAST_DEV_RUN = False
    CHANNELS=3
    CODEBOOK_SLOTS=512

    for i in range(LAYERS):
        print("Training layer", i)
        if i == 0:
            hqa = HQAGAN.init_bottom(
                input_feat_dim=CHANNELS,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_slots=CODEBOOK_SLOTS
            )
        else:
            hqa = HQAGAN.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                codebook_slots=CODEBOOK_SLOTS
            )
        logger = TensorBoardLogger("tb_logs", name="HQAGAN_Imagnet")
        if i > 0:
            trainer = pl.Trainer(max_epochs=20, logger=logger, devices=2, num_sanity_val_steps=0, fast_dev_run=FAST_DEV_RUN, strategy='ddp_find_unused_parameters_true')
        else:
            trainer = pl.Trainer(max_epochs=20, logger=logger, devices=2, num_sanity_val_steps=0, fast_dev_run=FAST_DEV_RUN)
        trainer.fit(model=hqa, train_dataloaders=dl_train, val_dataloaders=dl_test)
        if not FAST_DEV_RUN:
            trainer.save_checkpoint(f'./hqagan_imagenet_checkpoints/layer{i}.ckpt')
        hqa_prev = hqa


    