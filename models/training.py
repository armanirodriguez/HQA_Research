import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils import get_bit_usage, device, LAYER_NAMES, LOG_DIR, MODELS_DIR
from r_adam import RAdam
from scheduler import FlatCA
import os
from .hae import HQA
from torch.nn.utils import clip_grad_norm_


def decay_temp_linear(step, total_steps, temp_base, temp_min=0.001):
    factor = 1.0 - (step/total_steps)
    return temp_min + (temp_base - temp_min) * factor

def get_loss_hqa(img, model, epoch, step, commit_threshold=0.6, log=None):
    recon, orig, z_q, z_e, indices, KL, commit_loss = model(img)
    recon_loss = model.recon_loss(orig, recon)
    
    # calculate loss
    dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
    loss = recon_loss/dims 
    
    # logging    
    if step % 20 == 0:
        nll = recon_loss
        distortion_bpd = nll / dims / np.log(2)
        
        bits, max_bits, highest_prob = get_bit_usage(indices)
        bit_usage_frac = bits / max_bits
        
        time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_line = f"{time}, epoch={epoch}, step={step}, loss={loss:.5f}, distortion={distortion_bpd:.3f}, bit_usage={bit_usage_frac:.5f}, highest_prob={highest_prob:.3f}"
        print(log_line)

        if log is not None:
            with open(log, "a") as logfile:
                logfile.write(log_line + "\n")
                
    return loss, indices


def train(dl_train, test_x, model, optimizer, scheduler, epochs, decay=True, log=None):
    step = 0
    model.train()
    temp_base = model.codebook.temperature
    code_count = torch.zeros(model.codebook.codebook_slots).to(device)
    total_steps = len(dl_train)*epochs
    
    for epoch in range(epochs):
        for x, _ in dl_train:
            x = x.to(device)
            
            
            loss, indices = get_loss_hqa(x, model, epoch, step, log=log)
                
            # take training step    
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()     
                
            step += 1

def train_full_stack(dl_train, test_x, exp_name, epochs=5, lr=4e-4, layers = len(LAYER_NAMES)):
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    hqa = HQA.init_bottom(
        input_feat_dim=1,
        enc_hidden_dim=enc_hidden_sizes[0],
        dec_hidden_dim=dec_hidden_sizes[0],
    ).to(device)

    '''
    layers = LAYER_NAMES[:min(layers, len(LAYER_NAMES))]

    for i,_ in enumerate(layers):
        print(f"training layer{i}")
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=1,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
            ).to(device)
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
            ).to(device)
        
        print(f"layer{i} param count {sum(x.numel() for x in hqa.parameters()):,}")
        
        log_file = os.path.join(LOG_DIR, f"{exp_name}_l{i}.log")
        opt = RAdam(hqa.parameters(), lr=lr)
        scheduler = FlatCA(opt, steps=epochs*len(dl_train), eta_min=lr/10)
        train(dl_train, test_x, hqa, opt, scheduler, epochs, log=log_file)
        hqa_prev = hqa
    '''
    
    torch.save(hqa, f"{MODELS_DIR}/{exp_name}.pt")
    return hqa
