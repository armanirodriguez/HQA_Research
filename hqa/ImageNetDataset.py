from torch.utils.data import Dataset
from PIL import Image
import os
import json


class ImageNet100(Dataset):
    default_classes = [
        "n02869837",
        "n01749939",
        "n02488291",
        "n02107142",
        "n13037406",
        "n02091831",
        "n04517823",
        "n04589890",
        "n03062245",
        "n01773797",
        "n01735189",
        "n07831146",
        "n07753275",
        "n03085013",
        "n04485082",
        "n02105505",
        "n01983481",
        "n02788148",
        "n03530642",
        "n04435653",
        "n02086910",
        "n02859443",
        "n13040303",
        "n03594734",
        "n02085620",
        "n02099849",
        "n01558993",
        "n04493381",
        "n02109047",
        "n04111531",
        "n02877765",
        "n04429376",
        "n02009229",
        "n01978455",
        "n02106550",
        "n01820546",
        "n01692333",
        "n07714571",
        "n02974003",
        "n02114855",
        "n03785016",
        "n03764736",
        "n03775546",
        "n02087046",
        "n07836838",
        "n04099969",
        "n04592741",
        "n03891251",
        "n02701002",
        "n03379051",
        "n02259212",
        "n07715103",
        "n03947888",
        "n04026417",
        "n02326432",
        "n03637318",
        "n01980166",
        "n02113799",
        "n02086240",
        "n03903868",
        "n02483362",
        "n04127249",
        "n02089973",
        "n03017168",
        "n02093428",
        "n02804414",
        "n02396427",
        "n04418357",
        "n02172182",
        "n01729322",
        "n02113978",
        "n03787032",
        "n02089867",
        "n02119022",
        "n03777754",
        "n04238763",
        "n02231487",
        "n03032252",
        "n02138441",
        "n02104029",
        "n03837869",
        "n03494278",
        "n04136333",
        "n03794056",
        "n03492542",
        "n02018207",
        "n04067472",
        "n03930630",
        "n03584829",
        "n02123045",
        "n04229816",
        "n02100583",
        "n03642806",
        "n04336792",
        "n03259280",
        "n02116738",
        "n02108089",
        "n03424325",
        "n01855672",
        "n02090622",
    ]

    def __init__(self, root, split, transform=None, remap_labels=True):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.classes = self.default_classes
        if remap_labels:
            self.syn_to_class = {
                cls:index for index, cls in enumerate(self.classes)
            }
        else:
            self.syn_to_class = {}
            with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                json_file = json.load(f)
                for class_id, v in json_file.items():
                    self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
                if split == "train":
                    syn_id = entry
                    if syn_id in self.classes:
                        target = self.syn_to_class[syn_id]
                        syn_folder = os.path.join(samples_dir, syn_id)
                        for sample in os.listdir(syn_folder):
                            sample_path = os.path.join(syn_folder, sample)
                            self.samples.append(sample_path)
                            self.targets.append(target)
                elif split == "val":
                    syn_id = self.val_to_syn[entry]
                    if syn_id in self.classes:
                        target = self.syn_to_class[syn_id]
                        sample_path = os.path.join(samples_dir, entry)
                        self.samples.append(sample_path)
                        self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    DATA_FOLDER = '/home/dl_class/data/ILSVRC/Data/CLS-LOC/'
    dataset = ImageNet100(DATA_FOLDER, split='train', transform=T.Compose([
        T.CenterCrop(224),
        T.ToTensor()
    ]))
    dataloader = DataLoader(dataset, 64, True)
    for x, y in dataloader:
        print(y)
    
