from typing import List, Union

from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.datasets import Cars3DDataset, CelebA, MedleySolosDB, IRMAS, IEMOCAPXVectors


DATA_PATH = Path(__file__).resolve().parents[2] / "data"

def get_dataset(dataset, **kwargs):
    if dataset.upper()=='MNIST':
        return mnist(**kwargs)
    elif dataset.upper()=='SVHN':
        return svhn(**kwargs)
    elif dataset.upper()=='CARS3D':
        return cars3d(**kwargs)
    else:
        raise ValueError


#==========MNIST====================
def mnist(
        root: str=None,
        train: bool=True,
        desired_classes: Union[int, List[int]]=None,
    ):
    
    root = root or DATA_PATH / "MNIST"
    mean, std = [0.1307,], [0.3081,]
    
    if desired_classes is None:
        desired_classes = [i for i in range(10)]
    elif isinstance(desired_classes, int):
        desired_classes = [desired_classes]

    dataset = datasets.MNIST(
        root=root,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Pad(2),
        ]),
        download=True,
    )

    indices = torch.tensor([label in desired_classes for label in dataset.targets])
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
        
    return dataset, 1, 32, len(desired_classes), mean, std, True


#==========SVHN====================
def svhn(
        root: str=None,
        train: bool=True,
    ):
    
    root = root or DATA_PATH / "SVHN"
    mean = [0.4377, 0.4437, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    dataset = datasets.SVHN(
        root=root,
        split='train' if train else 'test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        download=True,
    )
        
    return dataset, 3, 32, 10, mean, std, True


#==========CARS 3D====================
def cars3d(
        root: str=None,
        train: bool=True,
        **kwargs
    ):
    
    root = root or DATA_PATH / "CARS3D"
    mean = [0.90318302, 0.89532674, 0.89166264]
    std = [0.24426176, 0.25783874, 0.26560575]

    dataset = Cars3DDataset(
        root=root,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        **kwargs
    )
        
    return dataset, 3, 128, dataset.n_classes, mean, std, True