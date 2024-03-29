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

from sklearn.manifold import TSNE

import os

# Mish Activation
def mish(x): return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

# FlatCA LR Scheduler
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

class VQCodebook(pl.LightningModule):
    def __init__(self, 
                 codebook_slots, 
                 codebook_dim, 
                 temperature=0.5):
        super().__init__()
        self.codebook_slots = codebook_slots
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.codebook = nn.Embedding(codebook_slots, codebook_dim) 
        self.log_slots_const = np.log(self.codebook_slots)

    def z_e_to_z_q(self, z_e, soft=True):
        bs, feat_dim, w, h = z_e.shape
        assert feat_dim == self.codebook_dim
        codebook = self.codebook.weight
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e.view(bs * w * h, feat_dim)
        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)

        distances = torch.addmm(
            codebook_sqr + z_e_flat_sqr, z_e_flat, codebook.t(), alpha=-2.0, beta=1.0
        )

        if soft is True:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, w, h)
            z_q = (soft_onehot @ codebook).view(bs, w, h, feat_dim)
            
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
                    .type_as(nn.Parameter(torch.randn(self.codebook_slots, self.codebook_dim).to(self.device)))
                    .view(bs * w * h, self.codebook_slots)
                )
                z_q = (hard_onehot @ codebook).view(bs, w, h, feat_dim)
                
                # entropy loss
                KL = dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))
                KL = KL.view(bs, w, h, self.codebook_slots).sum(dim=(1,2,3)).mean()

                commit_loss = 0.0

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, hard_indices, KL, commit_loss

    def lookup(self, ids: torch.Tensor):
        return self.codebook(ids).permute(0, 3, 1, 2)

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
    VISUALIZATION_DIR = 'vis'
    SUBDIRS=[VISUALIZATION_DIR]
    def __init__(
        self,
        input_feat_dim,
        prev_model=None,
        codebook_slots=256,
        codebook_dim=64,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        gs_temp=0.667,
        num_res_blocks=0,
        lr=4e-4,
        decay=True,
        clip_grads=False,
        codebook_init='uniform',
        output_dir=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])
        self.prev_model = prev_model
        self.encoder = Encoder(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks)
        self.codebook = VQCodebook(codebook_slots, codebook_dim, gs_temp)
        self.decoder = Decoder(
            codebook_dim,
            input_feat_dim,
            dec_hidden_dim,
            very_bottom=prev_model is None,
            num_res_blocks=num_res_blocks)
        
        self.normalize = GlobalNormalization(codebook_dim, scale=True)
        self.lr = lr
        self.decay = decay
        self.clip_grads = clip_grads

        # Tells pytorch lightinig to use our custom training loop
        self.automatic_optimization = False
        
        self.init_codebook(codebook_init)
        
        self.create_output = output_dir is not None 
        if self.create_output:
            self.output_dir = output_dir
            for subdir in HQA.SUBDIRS:
                path = os.path.join(output_dir, subdir, f'layer{len(self)}')
                os.makedirs(path, exist_ok=True)
    
    @torch.no_grad()
    def init_codebook(self, codebook_init):
        if codebook_init == 'uniform':
            self.codebook.codebook.weight.data.uniform_(-1./self.codebook.codebook_slots, 1./self.codebook.codebook_dim)
        elif codebook_init == 'normal':
            self.codebook.codebook.weight.data.normal_()
        else:
            raise Exception("Invalid codebook initialization")
            
    def on_train_start(self):
        # Register a buffer to track codeword usage
        self.register_buffer(
            'code_count', 
            torch.zeros(self.codebook.codebook_slots, device=self.device, dtype=torch.float64)
        )
        self.codebook_resets = 0
          
    def forward(self, x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_q)
        return z_e_lower_tilde, z_e_lower, z_q, z_e, indices, kl, commit_loss
    
    def get_training_loss(self, x):
        recon, _, _, _, indices, KL, commit_loss = self(x)
        recon_loss = self.recon_loss(x, recon)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = recon_loss/dims + 0.001*KL/dims + 0.001*(commit_loss)/dims

        return loss, indices, KL
    
    

    def recon_loss(self, orig, recon):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1,2,3)).mean()
    
    def decay_temp_linear(self, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step/total_steps)
        return temp_min + (temp_base - temp_min) * factor

    def training_step(self,batch, batch_idx):
        x,_ = batch
        
        # anneal temperature
        if self.decay:
                self.codebook.temperature = self.decay_temp_linear(step = self.global_step+1, 
                                                                   total_steps = self.trainer.max_epochs * self.trainer.num_training_batches, 
                                                                   temp_base= self.codebook.temperature)
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
    
        loss, indices, kl_loss = self.get_training_loss(x)
    
        optimizer.zero_grad()
        
        self.manual_backward(loss)
        
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

        indices_onehot = F.one_hot(indices, num_classes=self.codebook.codebook_slots).float()
        self.code_count = self.code_count + indices_onehot.sum(dim=(0, 1, 2))

        
        if batch_idx > 0 and batch_idx % 20 == 0:
            self.reset_least_used_codeword()
            if self.create_output:
                tsne = self.visualize_codebook()
                plt.clf()
                plt.scatter(tsne[:, 0], tsne[:, 1], marker='.', s=1)
                plt.title(f'Reset {self.codebook_resets} Epoch {self.current_epoch} Step {self.global_step} ')
                plt.savefig(f'{self.output_dir}/{HQA.VISUALIZATION_DIR}/layer{len(self)}/reset{self.codebook_resets}.png')
        
        
        self.log("loss", loss, prog_bar=True)
        self.log("kl", kl_loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_ndx):
        x,_ = batch
        y = self.reconstruct(x)
        loss = self.recon_loss(y,x)
        self.log("validation_recon_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = FlatCA(optimizer, steps=self.trainer.max_epochs * self.trainer.num_training_batches, eta_min=4e-5)
        return [optimizer], [lr_scheduler]
    
    @torch.no_grad()
    def reset_least_used_codeword(self):
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        frac_usage = self.code_count / max_count
        z_q_most_used = self.codebook.lookup(most_used_code.view(1, 1, 1)).squeeze()
        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        if min_frac_usage < 0.03:
            print(f'reset code {min_used_code}')
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / 100
            self.codebook.codebook.weight.data[min_used_code] = moved_code
        self.code_count = torch.zeros_like(self.code_count, device=self.device)
        self.codebook_resets += 1
    
    def visualize_codebook(self):
        """ Perform t-SNE visualization on the VQ-Codebook """
        latents = self.codebook.codebook.weight.data.detach().cpu().numpy()
        tsne = TSNE(n_components=2)
        latents_tsne = tsne.fit_transform(latents)
        return latents_tsne
        
    
    @torch.no_grad()
    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            z_e_lower = self.prev_model.encode(x)
            z_e_lower = self.normalize(z_e_lower)
        return z_e_lower
    
    @torch.no_grad()
    def encode(self, x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        return z_e
    
    @torch.no_grad()
    def decode_lower(self, z_q_lower):
       return self.prev_model.decode(z_q_lower)

    @torch.no_grad()
    def decode(self, z_q):
        if self.prev_model is not None:
            z_e_u = self.normalize.unnorm(self.decoder(z_q))
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
        model.prev_model.eval()
        return model
    
    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HQA(input_feat_dim,prev_model=None, **kwargs)
        return model

def train_mnist(n_epochs=20):
    torch.set_float32_matmul_precision('medium')
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )
    batch_size = 512
    ds_train = MNIST('./data/mnist', download=True, train=True, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    ds_test = MNIST('./data/mnist', download=True, train=False, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)

    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    for i in range(1):
        hqa_config = {
            'enc_hidden_dim':enc_hidden_sizes[i],
            'dec_hidden_dim':dec_hidden_sizes[i],
            'codebook_slots':512,
            'codebook_init':'uniform',
            'output_dir':'hqa_mnist_output'
        }
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=1,
                **hqa_config
            )
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                **hqa_config
            )
        logger = TensorBoardLogger("tb_logs", name="HQA_Mnist")
        trainer = pl.Trainer(max_epochs=n_epochs, logger=None, devices=2)
        
        trainer.fit(model=hqa, train_dataloaders=dl_train, val_dataloaders=dl_test)
        hqa.eval()
        hqa_prev = hqa
    return hqa
    
    
    
    
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    model = train_mnist(n_epochs=10)
    
        