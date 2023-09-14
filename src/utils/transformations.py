import numpy as np
import torch
import matplotlib.pyplot as plt


def tensor_to_numpy(input_tensor):
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.numpy()
    return input_tensor


def recover_img(img, mean=0, std=1):
    
    img = tensor_to_numpy(img)
    mean = tensor_to_numpy(mean)
    std = tensor_to_numpy(std)

    if len(img.shape) > 2:
        if img.shape[0]==1:
            img = img[0]
        else:
            img = img.transpose(1,2,0)

    img = img * std + mean

    img = (img-img.min())/(img.max()-img.min())

    img *= 255

    return img.astype(np.uint8)


def recover_spectrogram(spec, mean, std):
    
    spec = tensor_to_numpy(spec)
    mean = tensor_to_numpy(mean)
    std = tensor_to_numpy(std)
    
    spec = spec * std + mean
    spec = (spec-spec.min())/(spec.max()-spec.min())

    cm = plt.get_cmap('viridis')
    spec = cm(spec)[0,:,:,:3]

    spec *= 255
    return spec.astype(np.uint8)


def recover_tensor(input_tensor, mean=0, std=1, is_img=True):
    if is_img:
        return recover_img(input_tensor, mean, std)
    else:
        return recover_spectrogram(input_tensor, mean, std)