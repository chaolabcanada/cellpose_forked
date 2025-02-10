# inspiration from roi_detection3/dataset.py
import os
import numpy as np
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from pathlib import Path

from cellpose import io, transforms


class CellposeDataset(Dataset):
    """
    PyTorch Dataset for loading Cellpose training data.
    This dataset loads images and corresponding label files,
    applies necessary preprocessing (channel conversion, normalization),
    and optionally applies data augmentation.
    """
    def __init__(self, image_files, label_files, channels=None, channel_axis=None,
                 normalize_params={"normalize": False}, transform=None, phase='train'):
        """
        Args:
            image_files (list): List of file paths for training images.
            label_files (list): List of file paths for corresponding label files.
            channels (list or None): Indices of channels to keep.
            channel_axis (int or None): Axis that contains channel data.
            normalize_params (dict): Normalization parameters.
            transform (callable, optional): Transformation to apply (e.g., augmentation).
            phase (str): 'train' or 'test', affects augmentation.
        """
        if len(image_files) != len(label_files):
            raise ValueError("Number of images and labels must match.")
        self.image_files = image_files
        self.label_files = label_files
        self.channels = channels
        self.channel_axis = channel_axis
        self.normalize_params = normalize_params
        self.transform = transform  # TODO: add transformation function if we have any
        self.phase = phase


    def __getitem__(self, idx):
        img = io.imread(self.image_files[idx])
        label_tuple = io.imread_npy(self.label_files[idx])
        label = label_tuple[0]  # TODO: if label_tuple returns (masks, flows), we might choose one.

        if self.channels is not None or self.normalize_params.get("normalize", False):
            img = transforms.convert_image(img, channels=self.channels, channel_axis=self.channel_axis)
            img = img.transpose(2, 0, 1)  # ensure channel-first format
            if self.normalize_params.get("normalize", False):
                img = transforms.normalize_img(img, normalize=self.normalize_params, axis=0)

        if self.transform is not None:
            img_list, label_list = self.transform([img], [label])
            img, label = img_list[0], label_list[0]

        # convert image and label to torch.Tensor if they are not already.
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img)).float()
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label)).float()

        return img, label
    

    def __len__(self):
        return len(self.image_files)
    

def get_dataloader(image_files, label_files, batch_size=4, shuffle=True, num_workers=0,
                   channels=None, channel_axis=None, normalize_params={"normalize": False},
                   transform=None, phase='train'):
    """
    Factory function to create a DataLoader for the Cellpose dataset.
    
    Args:
        image_files (list): List of image file paths.
        label_files (list): List of label file paths.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of subprocesses for data loading.
        channels (list or None): Indices of channels to use.
        channel_axis (int or None): The axis index for channels.
        normalize_params (dict): Parameters for image normalization.
        transform (callable, optional): Augmentation transformation.
        phase (str): 'train' or 'test'.
    
    Returns:
        DataLoader: A PyTorch DataLoader wrapping the CellposeDataset.
    """
    dataset = CellposeDataset(
        image_files=image_files,
        label_files=label_files,
        channels=channels,
        channel_axis=channel_axis,
        normalize_params=normalize_params,
        transform=transform,
        phase=phase
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader