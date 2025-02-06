import logging
from typing import List, Tuple, Optional, Callable
import time
import os
from pathlib import Path
import random
import psutil
import numpy as np
import torch
from torch import nn
from tqdm import trange
from numba import prange
import asyncio

from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch
from cellpose.transforms import normalize_img


train_logger = logging.getLogger(__name__)


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
                       normalize_params={"normalize": False}):
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

def _normalize_probs(nimg):
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


def train_seg(net, train_files=None,
              train_labels_files=None, train_probs=None,
              test_files=None, test_labels_files=None,
              test_probs=None, batch_size=2, learning_rate=0.005,
              n_epochs=2000, weight_decay=1e-5, momentum=0.9, SGD=False, channels=None,
              channel_axis=None, normalize=True,
              save_path=None, save_every=100, rescale=True, scale_range=None, bsize=224,
              min_train_masks=5, model_name=None,
              nchan=2):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Boolean - whether to use SGD as optimization instead of RAdam. Defaults to False.
        channels (List[int], optional): List of ints - channels to use for training. Defaults to None.
        channel_axis (int, optional): Integer - axis of the channel dimension in the input data. Defaults to None.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.

    Returns:
        Path: path to saved model weights
    """
    # print(f"train_files: {len(train_files)}, type={type(train_files)}; train_labels_files: {len(train_labels_files)}")
    
    if not train_files or not train_labels_files:
        raise ValueError("train_files and train_labels_files must be provided.")
    else:
        nimg = len(train_files)
    
    if not test_files or not test_labels_files:
        raise ValueError("test_files and test_labels_files must be provided.")
    else:
        nimg_test = len(test_files)

    train_logger.info(f"nimg={nimg}, nimg_test={nimg_test}")

    device = net.device

    scale_range0 = 0.5 if rescale else 1.0
    scale_range = scale_range if scale_range is not None else scale_range0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    LR = learning_rate_scheduler(n_epochs, learning_rate=learning_rate)
    n_epochs = len(LR)

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        train_logger.info(
            f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
    else:
        train_logger.info(
            f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, momentum=momentum)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    # Save to a folder inside save path.
    save_path = os.getcwd() if save_path is None else save_path
    save_path = os.path.join(save_path, model_name)
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(save_path, f"{model_name}.pt")

    train_logger.info(f">>> saving model to {model_path}")

    lavg, nsum = 0, 0
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        net.train()
        
        # TODO: Properly skip batches when no images (after removing images with < min_train_masks)
        batches = get_batches_indices(train_files, batch_size)
        test_batches = get_batches_indices(test_files, batch_size)

        for inds in batches:
            # get file paths in batch
            print(f"\n\n\n\ninds: {inds}, {len(train_files)=}, {len(train_labels_files)=}")
            imgs_files, lbls_files = _get_batch(inds, files=train_files, labels_files=train_labels_files)
            train_logger.debug("Beginning processing training set.")

            # load files in batch into memory
            batch_outputs = _process_train_test(
                            train_files=imgs_files, train_labels_files=lbls_files,
                            test_files=None, test_labels_files=None, min_train_masks=min_train_masks,
                            channels=channels, channel_axis=channel_axis,
                            normalize_params=normalize_params, device=net.device,
                            nchan=nchan) # @floppase - add nchan option
            
            train_logger.debug("Training processing set done.")

            (imgs, lbls, curr_diam_train, _, _, _) = batch_outputs

            net.diam_labels.data = torch.Tensor([curr_diam_train.mean()]).to(device)

            train_logger.info(f"curr_diam_train.shape {curr_diam_train.shape}")
            diams = np.array([curr_diam_train[i] for i in list(range(len(inds)))])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            # augmentations
            imgi, lbl = transforms.random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                            scale_range=scale_range,
                                                            xy=(bsize, bsize))[:2]

            X = torch.from_numpy(imgi).to(device)
            train_logger.debug(f"X.shape - {X.shape}, lbl.shape {lbl.shape}")


            train_logger
            y = net(X)[0]

            # @floppase - Examine state of X, y
            train_logger.debug(f"X.shape - {X.shape}, y.shape - {y.shape}, lbl.shape {lbl.shape}")
            y_is_same = torch.all(y[:, 0, :, :] == y[:, 1, :, :]) and torch.all(y[:, 0, :, :] == y[:, 2, :, :])
            train_logger.debug(f"Y dimensions are the same: {y_is_same}")

            # return X, y, lbl

            loss = _loss_fn_seg(lbl, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_loss *= len(imgi)
            lavg += train_loss
            nsum += len(imgi)

        train_logger.info("Begin testing.")
        # testing
        if iepoch in [1, 2, 3, 4] or iepoch % 5 == 0:
            if test_files is not None:
                with torch.no_grad():
                    net.eval()

                    test_loss_avg, test_nsum = 0, 0

                    for inds in test_batches:
                        test_imgs_files, test_lbls_files = _get_batch(inds, files=test_files, labels_files=test_labels_files)

                        # load files in batch into memory
                        batch_outputs = _process_train_test(
                                        train_files=None, train_labels_files=None,
                                        test_files=test_imgs_files, test_labels_files=test_lbls_files, min_train_masks=min_train_masks,
                                        channels=channels, channel_axis=channel_axis,
                                        normalize_params=normalize_params, device=net.device,
                                        nchan=nchan)

                        (_, _, _, test_imgs, test_lbls, curr_diam_test) = batch_outputs

                        test_diams = np.array([x for x in curr_diam_test])
                        train_logger.info("Generated %d size array", test_diams.nbytes)
                        test_rsc = test_diams / net.diam_mean.item() if rescale else np.ones(
                            len(test_diams), "float32")
                        
                        test_imgi, test_lbl = transforms.random_rotate_and_resize(
                            test_imgs, Y=test_lbls, rescale=test_rsc, scale_range=scale_range,
                            xy=(bsize, bsize))[:2]

                        test_X = torch.from_numpy(test_imgi).to(device)
                        test_y = net(test_X)[0]
                        test_loss = _loss_fn_seg(test_lbl, test_y, device).item()

                        test_loss_avg += test_loss * len(inds)
                        test_nsum += len(inds)

                    train_logger.info("Computed test loss: %.2f", (test_loss / test_nsum))
            print(f"lavg={lavg}, nsum={nsum}")
            lavg /= nsum
            train_logger.info(
                f"epoch={iepoch}, train_loss={lavg:.4f}, test_loss={test_loss:.4f}, LR={LR[iepoch]:.4f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        train_logger.info("End of epoch %d. Free memory: %.2f GB", iepoch, (psutil.virtual_memory().available / 1e9))

        if iepoch > 0 and iepoch % save_every == 0:
            net.save_model(os.path.join(save_path, f"{model_name}_{iepoch}.pt"))
    net.save_model(model_path)

    return model_path, test_X, test_y, test_lbl

def train_size(net, pretrained_model, train_data=None, train_labels=None,
               train_files=None, train_labels_files=None, train_probs=None,
               test_data=None, test_labels=None, test_files=None,
               test_labels_files=None, test_probs=None, load_files=True,
               min_train_masks=5, channels=None, channel_axis=None, rgb=False,
               normalize=True, nimg_per_epoch=None, nimg_test_per_epoch=None,
               batch_size=64, scale_range=1.0, bsize=512, l2_regularization=1.0,
               n_epochs=10):
    """Train the size model.

    Args:
        net (object): The neural network model.
        pretrained_model (str): The path to the pretrained model.
        train_data (numpy.ndarray, optional): The training data. Defaults to None.
        train_labels (numpy.ndarray, optional): The training labels. Defaults to None.
        train_files (list, optional): The training file paths. Defaults to None.
        train_labels_files (list, optional): The training label file paths. Defaults to None.
        train_probs (numpy.ndarray, optional): The training probabilities. Defaults to None.
        test_data (numpy.ndarray, optional): The test data. Defaults to None.
        test_labels (numpy.ndarray, optional): The test labels. Defaults to None.
        test_files (list, optional): The test file paths. Defaults to None.
        test_labels_files (list, optional): The test label file paths. Defaults to None.
        test_probs (numpy.ndarray, optional): The test probabilities. Defaults to None.
        load_files (bool, optional): Whether to load files. Defaults to True.
        min_train_masks (int, optional): The minimum number of training masks. Defaults to 5.
        channels (list, optional): The channels. Defaults to None.
        channel_axis (int, optional): The channel axis. Defaults to None.
        normalize (bool or dict, optional): Whether to normalize the data. Defaults to True.
        nimg_per_epoch (int, optional): The number of images per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): The number of test images per epoch. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 64.
        l2_regularization (float, optional): The L2 regularization factor. Defaults to 1.0.
        n_epochs (int, optional): The number of epochs. Defaults to 10.

    Returns:
        dict: The trained size model parameters.
    """
    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data, train_labels=train_labels, train_files=train_files,
        train_labels_files=train_labels_files, train_probs=train_probs,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        test_labels_files=test_labels_files, test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks, compute_flows=False,
        channels=channels, channel_axis=channel_axis, normalize_params=normalize_params,
        device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out

    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    diam_mean = net.diam_mean.item()
    device = net.device
    net.eval()

    styles = np.zeros((n_epochs * nimg_per_epoch, 256), np.float32)
    diams = np.zeros((n_epochs * nimg_per_epoch,), np.float32)
    tic = time.time()
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for ibatch in range(0, nimg_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_per_epoch, ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, **kwargs)
            diami = diam_train[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            with torch.no_grad():
                feat = net(imgi)[1]
            indsi = inds_batch + nimg_per_epoch * iepoch
            styles[indsi] = feat.cpu().numpy()
            diams[indsi] = np.log(diami) - np.log(diam_mean) + np.log(scale)
        del feat
        train_logger.info("ran %d epochs in %0.3f sec" %
                          (iepoch + 1, time.time() - tic))

    l2_regularization = 1.

    # create model
    smean = styles.copy().mean(axis=0)
    X = ((styles.copy() - smean).T).copy()
    ymean = diams.copy().mean()
    y = diams.copy() - ymean

    A = np.linalg.solve(X @ X.T + l2_regularization * np.eye(X.shape[0]), X @ y)
    ypred = A @ X

    train_logger.info("train correlation: %0.4f" % np.corrcoef(y, ypred)[0, 1])

    if nimg_test:
        np.random.seed(0)
        styles_test = np.zeros((nimg_test_per_epoch, 256), np.float32)
        diams_test = np.zeros((nimg_test_per_epoch,), np.float32)
        diams_test0 = np.zeros((nimg_test_per_epoch,), np.float32)
        if nimg_test != nimg_test_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg_test),
                                     size=(nimg_test_per_epoch,), p=test_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg_test))
        for ibatch in range(0, nimg_test_per_epoch, batch_size):
            inds_batch = np.arange(ibatch, min(nimg_test_per_epoch,
                                               ibatch + batch_size))
            inds = rperm[inds_batch]
            imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels,
                                    files=test_files, labels_files=test_labels_files,
                                    **kwargs)
            diami = diam_test[inds].copy()
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgs, Y=lbls, scale_range=scale_range, xy=(bsize, bsize))
            imgi = torch.from_numpy(imgi).to(device)
            diamt = np.array([utils.diameters(lbl0[0])[0] for lbl0 in lbl])
            diamt = np.maximum(5., diamt)
            with torch.no_grad():
                feat = net(imgi)[1]
            styles_test[inds_batch] = feat.cpu().numpy()
            diams_test[inds_batch] = np.log(diami) - np.log(diam_mean) + np.log(scale)
            diams_test0[inds_batch] = diamt

        diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(diam_mean) + ymean)
        diam_test_pred = np.maximum(5., diam_test_pred)
        train_logger.info("test correlation: %0.4f" %
                          np.corrcoef(diams_test0, diam_test_pred)[0, 1])

    pretrained_size = str(pretrained_model) + "_size.npy"
    params = {"A": A, "smean": smean, "diam_mean": diam_mean, "ymean": ymean}
    np.save(pretrained_size, params)
    train_logger.info("model saved to " + pretrained_size)

    return params
