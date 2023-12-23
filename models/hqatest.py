import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from training import train_full_stack
from utils import *
from torchvision.utils import make_grid
import numpy as np
from DL_class.intro_AE.hae import *
import pandas as pd
from load_datasets import *

def main(model_name, model_save_path, img_save_dir, dl_train, dl_test, layers = 5):
    
    print(f"CUDA={torch.cuda.is_available()}", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
        #SAMPLE MNIST DATASETS
    test_x, _ = next(iter(dl_test))
    test_x = test_x.to(device)

    
    # Train a AE stack
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    
    if not os.path.isfile(model_save_path):
        hqa_model = train_full_stack(dl_train, test_x, model_name, epochs=30, layers = layers)
    else:
        hqa_model = torch.load(model_save_path)
    
    hqa_model.eval()
        
    layer_descriptions = [
        "downsample 2 in each dimension, latent space size of 16x16",
        "downsample 4 in each dimension, latent space size of 8x8",
        "downsample 8 in each dimension, latent space size of 4x4",
        "downsample 16 in each dimension, latent space size of 2x2",
        "downsample 32 in each dimension, latent space size of 1x1",
    ]

    # Show reconstruction comparison over each layer in HQA
    recon_comparison(hqa_model, dl_test.dataset, LAYER_NAMES, layer_descriptions, img_save_dir)
    
    
    
    ds_test = dl_test.dataset
    # Final layer interpolations
    
    grid_x = grid_y = 16
    results = torch.Tensor(grid_x * grid_y, 1, 32, 32)
    i = 0
    
    for j in range(grid_y):
        x_a,_ = ds_test[j]
        x_b,_ = ds_test[j+grid_y]
        point_1 = hqa_model.encode(x_a.unsqueeze(0).to(device)).cpu()
        point_2 = hqa_model.encode(x_b.unsqueeze(0).to(device)).cpu()
        interpolate_x = np.linspace(point_1[0], point_2[0], grid_x)
    
        for z_e_interpolated in interpolate_x:
            z_e_i = torch.Tensor(z_e_interpolated).unsqueeze(0).to(device)
            z_q = z_e_i #hqa_model.quantize(z_e_i)
            results[i] = hqa_model.decode(z_q).squeeze()
            i += 1
                
    grid_img = make_grid(results.cpu(), nrow=grid_x)
    #show_image(grid_img[0,:,:])
    
    #Stochastic Reconstructions
    
    # Show held-out reconstructions: [ORIG, 14xSAMPLE, AVERAGED_10_SAMPLES]
    grid_x = grid_y = 16
    results = torch.Tensor(grid_x * grid_y, 1, 32, 32)
    
    result_idx = 0
    for test_idx in range(grid_y):
        x_a,_ = ds_test[test_idx]
        img = x_a.squeeze().to(device)
        img_ = img.unsqueeze(0).unsqueeze(0)
        num_examples = 5
        
        # ORIG
        results[result_idx] = img
        result_idx += 1
        
        # 14 RANDOM STOCHASTIC DECODES
        for _ in range(grid_x -2):
            results[result_idx] = hqa_model.reconstruct(img_).squeeze()
            result_idx += 1
        
        # AVERAGED SAMPLES
        results[result_idx] = hqa_model.reconstruct_average(img_, num_samples=14).squeeze()
        result_idx += 1
    
    grid_img = make_grid(results.cpu(), nrow=grid_x)
    #show_image(grid_img[0,:,:])
    
    # Layer-wise interpolations
    print("Originals")
    show_original(1, ds_test)
    show_original(9, ds_test)
    for layer, name, description in zip(hqa_model, LAYER_NAMES, layer_descriptions):
        print(f"{name} : {description}")
        interpolate(1, 9, ds_test, layer, grid_x=10)
        

def compress_MNIST_datasets():

    dl_train, _, dl_test = load_mnist(validate=False)
    main(HQA_MNIST_MODEL_NAME,
        HQA_MNIST_SAVE_PATH,
        IMG_MNIST_DIR_PATH,
        dl_train,
        dl_test)



if __name__ == "__main__":
    set_seeds()
    compress_MNIST_datasets()
