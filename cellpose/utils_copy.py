import logging
from typing import List, Tuple, Optional, Callable
import time
import os
from pathlib import Path
import random
import psutil # type: ignore
import numpy as np
import torch # type: ignore
from torch import nn # type: ignore
from tqdm import trange
from numba import prange # type: ignore
import asyncio

from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch
from cellpose.transforms import normalize_img

train_logger = logging.getLogger(__name__)


# Loss Function below
def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * torch.from_numpy(lbl[:, 1:]).to(device)
    loss = criterion(y[:, :2], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], torch.from_numpy(lbl[:, 0] > 0.5).to(device).float())
    loss = loss + loss2
    return loss


# Batch processing functions below
def _get_batch(inds, files: list, labels_files: list):
    """
    Gets a batch of file paths.

    Args:
        inds (list): List of indices indicating which file paths to retrieve.
        files (list): List of image file paths.
        labels_files (list): List of label file paths.

    Returns:
        tuple: A tuple containing two lists: the batch of image file paths and the batch of label file paths.
    """
    if files is None:
        raise ValueError("Image file paths must be provided.")

    batch_files = [files[i] for i in inds]
    batch_labels_files = [labels_files[i] for i in inds] if labels_files else None

    return batch_files, batch_labels_files


def get_batches_indices(file_list: List[str], batch_size: int) -> List[List[int]]:
    file_indices = list(range(len(file_list)))
    train_logger.debug(f"File indices: {file_indices}")
    # randomly shuffle file_indices
    random.shuffle(file_indices)

    batches = []

    while file_indices:
        if len(file_indices) < batch_size:
            batches.append(file_indices)
            file_indices = None
        else:
            curr_batch = file_indices[:batch_size]
            file_indices = file_indices[batch_size:]

            batches.append(curr_batch)
    
    train_logger.debug(f"Current batches: {batches} (files: {len(file_list)})")
    train_logger.debug(f"Batch sizes: {[len(batch) for batch in batches]}")
    train_logger.debug(f"Total batches: {len(batches)}")

    return batches


# Image processing functions below
def pad_to_rgb(img):
    if img.ndim == 2 or np.ptp(img[1]) < 1e-3:
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        img = np.tile(img[:1], (3, 1, 1))
    elif img.shape[0] < 3:
        nc, Ly, Lx = img.shape
        # randomly flip channels
        if np.random.rand() > 0.5:
            img = img[::-1]
        # randomly insert blank channel
        ic = np.random.randint(3)
        img = np.insert(img, ic, np.zeros((3 - nc, Ly, Lx), dtype=img.dtype), axis=0)
    return img


def convert_to_rgb(img):
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        img = np.tile(img, (3, 1, 1))
    elif img.shape[0] < 3:
        img = img.mean(axis=0, keepdims=True)
        img = transforms.normalize99(img)
        img = np.tile(img, (3, 1, 1))
    return img


def _reshape_norm(data, channels=None, channel_axis=None, rgb=False,
                  normalize_params={"normalize": False}, nchan=2):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data.
        channels (int or list, optional): Number of channels or list of channel indices to keep. Defaults to None.
        channel_axis (int, optional): Axis along which the channels are located. Defaults to None.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if channels is not None or channel_axis is not None:
        data = [
            transforms.convert_image(td, channels=channels, channel_axis=channel_axis, nchan=nchan)
            for td in data
        ]
        data = [td.transpose(2, 0, 1) for td in data]
    if normalize_params["normalize"]:
        data = [
            transforms.normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    if rgb:
        data = [pad_to_rgb(td) for td in data]
    return data


def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}): # Unused
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = transforms.convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = transforms.normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


# File handling functions
async def imread_batch_async(files: List[str], async_imread_func: Callable) -> List[np.ndarray]:
    if async_imread_func not in [io.imread_async, io.imread_npy_async]:
        raise ValueError("Invalid async imread function")

    try:
        results = await asyncio.gather(*[async_imread_func(f) for f in files])
    except Exception as e:
        raise RuntimeError(f"Error reading batch of files: {e}")

    return results


def _load_files(files: List[str], labels_files: List[str], load_mode: Optional[str] = 'async') -> Tuple[List[np.ndarray], List[np.ndarray], List[list]]:
    """
    Load image and label files.

    Args:
        files (List[str]): List of paths to image files.
        labels_files (List[str]): List of paths to label files.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[list]]: List of loaded images, labels and the corresponding flows of the labels.
    """
    nimg = len(files)
    data = [io.imread(files[i]) for i in trange(nimg)]
    
    # use function which gives _npy format priority (load as data dict with "masks" and "flows" as keys instead of masks only)
    # zip(*) is used to unpack list of tuples into two lists

    if load_mode == 'async':
        gather_labels = asyncio.run(imread_batch_async(labels_files, async_imread_func=io.imread_npy_async))
        labels, labels_flows = zip(*gather_labels)
    else:
        # Non-async code... (deprecated)
        labels, labels_flows = zip(*[io.imread_npy(labels_files[i]) for i in trange(nimg)])

    if labels is not None and nimg != len(labels):
        train_logger.critical("Data and labels not same length")
        raise ValueError
    if labels is not None:
        if labels[0].ndim < 2 or data[0].ndim < 2:
            train_logger.critical("Data or labels are not at least two-dimensional")
            raise ValueError
        if data[0].ndim > 3:
            train_logger.critical("Data is more than three-dimensional (should be 2D or 3D array)")
            raise ValueError
    
    return data, labels, labels_flows


# Label processing functions
def _compute_diameters(labels):
    """
    Compute diameters for labels.

    Args:
        labels: Labels to compute diameters for.

    Returns:
        Computed diameters.
    """
    nimg = len(labels)
    nmasks = np.zeros(nimg)
    diam = np.zeros(nimg)
    for k in trange(nimg):
        tl = labels[k][0]
        diam[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam[diam<5] = 5.
    return diam, nmasks


def _normalize_probs(nimg): # Unused
    """
    Normalize probabilities.

    Args:
        nimg: Number of images.

    Returns:
        Normalized probabilities.
    """
    probs = 1./nimg * np.ones(nimg, "float64")
    probs /= probs.sum()
    return probs


def _process_train_test(train_files: Optional[List[str]], 
                        train_labels_files: Optional[List[str]],
                        test_files: Optional[List[str]] = None, 
                        test_labels_files: Optional[List[str]] = None, 
                        min_train_masks: int = 5,
                        channels: Optional[List[int]] = None, 
                        channel_axis: Optional[int] = None,
                        normalize_params: dict = {"normalize": False},
                        nchan: int = 2,
                        device: torch.device = torch.device("cuda")) -> Tuple:
    """
    Process training and testing data.

    Args:
        train_files (Optional[List[str]]): List of paths to training image files.
        train_labels_files (Optional[List[str]]): List of paths to training label files.
        test_files (Optional[List[str]]): List of paths to testing image files.
        test_labels_files (Optional[List[str]]): List of paths to testing label files.
        min_train_masks (int): Minimum number of training masks.
        channels (Optional[List[int]]): List of channels.
        channel_axis (Optional[int]): Channel axis.
        normalize_params (dict): Normalization parameters.
        device (torch.device): Device to use for computations.

    Returns:
        Tuple of training and testing data, labels, and flows
    """
    # Load files
    load_mode = 'async'

    # Note that flows that are not valid will be returned as None (in the list)
    
    if train_files and train_labels_files:
        
        train_logger.debug(">>> loading images and labels")
        t0 = time.time()
        train_data, train_labels, train_labels_flows = _load_files(train_files, train_labels_files, load_mode=load_mode)
        nimg = len(train_data)
        train_logger.debug(f">>> loaded files {nimg} in {time.time()-t0:.2f}s ({load_mode=})")
    else:
        train_data, train_labels = None, None
    if test_files and test_labels_files:
        test_data, test_labels, test_labels_flows = _load_files(test_files, test_labels_files, load_mode=load_mode)
        nimg_test = len(test_data)
        train_logger.debug(f">>> loaded files {nimg_test}")
    else:
        test_data, test_labels = None, None
    
    ## Check if flows are included within the labels before computing (avoid computing twice)
    

    ## Compute flows
    if train_files and train_labels_files:
        train_logger.debug(">>> computing flows")

        # if train_labels_flows is list and all items are not None - do not compute flows
        if isinstance(train_labels_flows, (list, tuple)) and all([x is not None for x in train_labels_flows]):
            train_labels = train_labels_flows
            print(">>> using precomputed flows")
        else:
            print(f">>> flows not precomputed, must compute flows {type(train_labels_flows)=}\n {len(train_labels_flows)=}")
            train_labels = dynamics.labels_to_flows(train_labels, device=device)
            
        
        # subset labels to remove first frame
        train_labels = [label[1:] for label in train_labels]
    else:
        train_labels = None
    
    if test_labels is not None:
        # if test_labels_flows is list and all items are not None - do not compute flows
        if isinstance(test_labels_flows, list) and all([x is not None for x in test_labels_flows]):
            test_labels = test_labels_flows
        else:
            test_labels = dynamics.labels_to_flows(test_labels, device=device)

        # subset labels to remove first frame (instance labels)
        # 0 = instance masks; 1 = masks; 2 = flows, 3 = flows
        test_labels = [label[1:] for label in test_labels]
    else:
        test_labels = None

    ### compute diameters
    if train_labels is not None:
        train_logger.debug(">>> computing diameters")
        diam_train, nmasks = _compute_diameters(train_labels)
    else:
        diam_train = None
    if test_labels is not None:
        diam_test, _ = _compute_diameters(test_labels)
    else:
        diam_test = None
    
    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set")
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
            if train_labels is not None:
                train_labels = [train_labels[i] for i in ikeep]
            diam_train = diam_train[ikeep]

    ### reshape and normalize train / test data
    if channels is not None or normalize_params["normalize"]:
        if channels:
            train_logger.debug(f">>> using channels {channels}")
        if normalize_params["normalize"]:
            if train_data is not None:
                train_logger.debug(f">>> normalizing {normalize_params}")
                train_data = _reshape_norm(train_data, channels=channels, 
                                channel_axis=channel_axis, normalize_params=normalize_params,
                                nchan=nchan)
            if test_data is not None:
                test_data = _reshape_norm(test_data, channels=channels, 
                                channel_axis=channel_axis, normalize_params=normalize_params,
                                nchan=nchan)
        
    return (train_data, train_labels, diam_train, 
            test_data, test_labels, diam_test)


# learning rate scheduler
def learning_rate_scheduler(n_epochs: int, learning_rate: float = 0.005):
    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 100:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    
    return LR