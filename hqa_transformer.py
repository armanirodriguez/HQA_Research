import lightning.pytorch as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from models.hqa_lightning import HQA
from gpt import GPT

def load_heirarchical_model_from_checkpoints(
    model_class,
    checkpoints_dir,
    n_layers=5,
    **kwargs
) -> pl.LightningModule:
    """
    Load a Heirarchical model from a directory of checkpoints for each layer
    """
    for layer in range(n_layers):
        if layer == 0:
            hae = model_class.load_from_checkpoint(
                f"{checkpoints_dir}/layer{layer}.ckpt", prev_model=None, **kwargs
            ).eval()
        else:
            hae = model_class.load_from_checkpoint(
                f"{checkpoints_dir}/layer{layer}.ckpt", prev_model=hae_prev, **kwargs
            ).eval()
        hae_prev = hae
    return hae

class HQATransformer(pl.LightningModule):
    def __init__(self, hqa_model, keep_ratio=0.5):
        super(HQATransformer, self).__init__()
        self.hqa_model = hqa_model
        self.vocab_size = hqa_model.codebook.codebook_slots
        self.gpt = GPT(
            vocab_size = self.vocab_size,
            block_size = 512,
            n_layer = 24,
            n_head = 16,
            n_embd = 1024
        )
        self.keep_ratio = keep_ratio
    
    @torch.no_grad()
    def quantize(self, x):
        z_e_lower = self.hqa_model.encode_lower(x)
        z_e = self.hqa_model.encoder(z_e_lower)
        z_q, indices, _, _ = self.hqa_model.codebook.z_e_to_z_q(z_e, soft=False)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices
    
    def forward(self, x):
        batch_size = x.shape[0]
        _, indices = self.quantize(x)
        
        sos_tokens = torch.zeros(batch_size, 1).long().to(self.device)
        
        mask = torch.bernoulli(self.keep_ratio * torch.ones(indices.shape, device=self.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        
        logits, _ = self.gpt(new_indices[:, :-1])
        
        return logits, indices
    
    def training_step(self, batch, batch_index):
        x, _ = batch
        logits, targets = self(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.gpt.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.gpt.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

if __name__ == '__main__':
    transform = T.Compose([
        T.CenterCrop(32),
        T.ToTensor()
    ])
    ds_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    dl_train = DataLoader(ds_train, 
                          shuffle=True, 
                          batch_size=32, 
                          num_workers=32)
    hqa_trainer = pl.Trainer(max_epochs=20, devices=[1])
    hqa_model = HQA.init_bottom(
                input_feat_dim=1,
                enc_hidden_dim=16,
                dec_hidden_dim=16,
                codebook_slots=512)
    hqa_trainer.fit(hqa_model, dl_train)
    
    hqa_transformer = HQATransformer(hqa_model)
    hqa_transformer_trainer = pl.Trainer(max_epochs=20, devices=[1])
    hqa_transformer_trainer.fit(hqa_transformer, dl_train)
    
    


        
        