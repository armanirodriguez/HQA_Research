import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from train_hae import load_hae_from_checkpoints, get_dataloader
from hqa.hqa_lightning import HQA

device = 'cuda:1'

def load_heirarchical_model_from_checkpoints(
    model_class,
    checkpoints_dir,
    n_layers=5,
    **kwargs
) -> pl.LightningModule:
    """
    Load a HAE model from a directory of checkpoints for each layer
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


if __name__ == '__main__':
    N_LAYERS = 5
    dl_train = get_dataloader()
    dl_test = get_dataloader(training=False)
    hqa = load_heirarchical_model_from_checkpoints(HQA, './hqa/hqa_checkpoints').to(device)
    classifier = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    total, correct = 0, 0
    for x, y in tqdm(dl_test):
        x = x.to(device)
        batch_size = x.shape[0]
        total += batch_size
        y_pred = classifier(x).to('cpu')
        correct += torch.sum(y == torch.argmax(y_pred, dim=1)).item()
    accuracy = correct / total
    print("Orignal data")
    print(f"{correct}/{total} correct ({accuracy} accuracy)")

    original_accuracies = np.full(N_LAYERS, accuracy)
    hae_accuracies = []

    for index, layer in enumerate(hqa):
        total, correct = 0, 0
        for x, y in tqdm(dl_test):
            x = x.to(device)
            x = layer.reconstruct(x)
            batch_size = x.shape[0]
            total += batch_size
            y_pred = classifier(x).to('cpu')
            correct += torch.sum(y == torch.argmax(y_pred, dim=1)).item()
        accuracy = correct/total
        print("Layer", index)
        print(f"{correct}/{total} correct ({accuracy} accuracy)")
        hae_accuracies.append(accuracy)
    
    hae_accuracies = np.array(hae_accuracies)

    plt.figure(figsize=(10,10))

    plt.plot(original_accuracies, label="Original Dataset")
    plt.plot(hae_accuracies, label="HQA Reconstructions")

    plt.xticks(np.arange(0, N_LAYERS))
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.savefig('hae_vs_original.png')
    

