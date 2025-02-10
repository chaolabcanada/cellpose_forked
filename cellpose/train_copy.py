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
import utils_copy # changes

from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch
from cellpose.transforms import normalize_img


train_logger = logging.getLogger(__name__)


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

    LR = utils_copy.learning_rate_scheduler(n_epochs, learning_rate=learning_rate)
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
        batches = utils_copy.get_batches_indices(train_files, batch_size)
        test_batches = utils_copy.get_batches_indices(test_files, batch_size)

        for inds in batches:
            # get file paths in batch
            print(f"\n\n\n\ninds: {inds}, {len(train_files)=}, {len(train_labels_files)=}")
            imgs_files, lbls_files = utils_copy._get_batch(inds, files=train_files, labels_files=train_labels_files)
            train_logger.debug("Beginning processing training set.")

            # load files in batch into memory
            batch_outputs = utils_copy._process_train_test(
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

            loss = utils_copy._loss_fn_seg(lbl, y, device)
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
                        test_imgs_files, test_lbls_files = utils_copy._get_batch(inds, files=test_files, labels_files=test_labels_files)

                        # load files in batch into memory
                        batch_outputs = utils_copy._process_train_test(
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
                        test_loss = utils_copy._loss_fn_seg(test_lbl, test_y, device).item()

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

    out = utils_copy._process_train_test(
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
            imgs, lbls = utils_copy._get_batch(inds, data=train_data, labels=train_labels,
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
            imgs, lbls = utils_copy._get_batch(inds, data=test_data, labels=test_labels,
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
