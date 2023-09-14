import os

import scipy.io
from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset


class Cars3DDataset(VisionDataset):

    def __init__(
            self, 
            root, 
            train,
            angles=None,
            elevations=None, 
            transform = None,
            target_transform = None
        ):
        
        root = str(root)
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        angles = list(range(24)) if angles is None else angles
        elevations = [0,1,2,3] if elevations is None else elevations
        
        files = [int(file.split("_")[1]) for file in os.listdir(root) if file.endswith(".mat")]
        cars = {i: scipy.io.loadmat(root+"/car_{:03d}_mesh.mat".format(i)) for i in files}
        self.n_classes = len(angles) * len(elevations)
        self.data = np.array([
            cars[i]['im'][:,:,:,j,k].transpose(2,0,1) for i in cars for j in angles for k in elevations
        ])
        self.labels = np.array([
            np.int64(len(angles)*k+j) for _ in cars for j in angles for k in range(len(elevations))
        ])
        self._split_dataset(train)


    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
    
    
    def _split_dataset(self, train):
        np.random.seed(1234)
        choice = np.random.choice(
            range(self.data.shape[0]), size=(self.data.shape[0]//4,), replace=False
        ) 
        test_idxs = np.zeros(self.data.shape[0], dtype=bool)
        test_idxs[choice] = True
        train_idxs = ~test_idxs
        if train:
            self.data = self.data[train_idxs]
            self.labels = self.labels[train_idxs]
        else:
            self.data = self.data[test_idxs]
            self.labels = self.labels[test_idxs]