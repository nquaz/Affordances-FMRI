from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from nilearn.image import resample_to_img
from numpy._typing import ArrayLike
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoModelForCausalLM
from torchvision.models import resnet50
import torch
from PIL import Image
from nilearn import image
import os
from os.path import join
import shutil
from tqdm import tqdm
import torchvision.transforms as T
from nilearn.plotting import  plot_glass_brain
from scipy.stats import pearsonr, zscore, spearmanr
from scipy.spatial.distance import squareform, pdist
from nilearn.masking import apply_mask, unmask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

from r3m import load_r3m
import pandas as pd

## TEXT UTILS
def get_single_string_embedding(my_string, tokenizer, embedding_layer):
    """Returns the embedding of a single string by averaging the embeddings of its tokens"""
    assert len(my_string) > 0, "Input string is empty"
    assert type(my_string) == str, "Input should be a string"
    tokens = tokenizer(my_string, add_special_tokens=False)['input_ids']
    return embedding_layer(torch.tensor(tokens)).detach().cpu().numpy().mean(0)


def get_multi_string_embedding(my_strings, tokenizer, embedding_layer):
    """Takes in a list of two strings, and returns a single embedding for both jointly by averaging passing them in one order or the other"""
    #FIXME it seems like the way it's being loaded/used now, BERT isn't actually using positional embedding so this isn't strictly necessary
    assert len(my_strings) == 2, "This should only take two strings"
    original_order_str = str.join(' ', my_strings)
    reverse_order_str = str.join(' ', my_strings[::-1])
    original_order_embedding = get_single_string_embedding(original_order_str, tokenizer, embedding_layer)
    reverse_order_embedding = get_single_string_embedding(reverse_order_str, tokenizer, embedding_layer)
    return (original_order_embedding + reverse_order_embedding)/2


def int_to_color(val, vmin=0, vmax=100, cmap_name='viridis'):
    """
    Map integer to color

    Args:
        val:
        vmin:
        vmax:
        cmap_name:

    Returns:

    """
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm(val))


def load_fmri(fmri_dir: str, brainmask_img: nib.nifti1.Nifti1Image, run_name_template: str, all_run_numbers: list[int], ROI: str=None, img: bool = False) -> dict[int, ArrayLike]:
    """
    Load fMRI from specified directory and list of runs

    Args:
        fmri_dir: directory where fmri files reside
        brainmask_img: mask for separating the brain from background
        run_name_template: filename template for each run
        all_run_numbers: list of runs
        ROI: region of interest. If `whole`, whole brain voxels are returned. Otherwise, specify region from Harvard-Oxford atlas and only voxels from that region will be returned.
        img: if True, also load the volumetric masked data

    Returns:
        dictionary mapping runs to 2D array of voxels of shape (# TRs, # voxels)
    """
    runs_fmri = {}
    runs_img = {}
    if ROI is None:
        ROI = 'whole'
    if ROI != 'whole':
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

    for run_number in tqdm(all_run_numbers):

        # get the run path
        run_path = join(fmri_dir, run_name_template.format(run_number))

        # read the run img
        run_img = image.load_img(run_path)

        if ROI == 'whole':
            # mask out non brain voxels
            masked_data = apply_mask(run_img, brainmask_img)
            runs_fmri[run_number] = masked_data

            if img:
                # Reconstruct the 4D volumetric masked data
                masked_img = unmask(masked_data, brainmask_img)
                runs_img[run_number] = masked_img
        else:
            # Get atlas labels
            resampled_atlas_img = resample_to_img(atlas.maps, run_img, interpolation='nearest', force_resample=True,
                                                  copy_header=True)
            atlas_data = resampled_atlas_img.get_fdata()

            # Group voxels by target labels
            index = atlas.labels.index(ROI)

            # Create 3D boolean mask
            region_mask = atlas_data == index  # shape (x, y, z)

            run_data = run_img.get_fdata().copy()
            runs_fmri[run_number] = run_data[region_mask, :].copy().T

            if img:
                # Save 4D Nifti for ease of plotting
                run_data[~region_mask, :] = 0
                runs_img[run_number] = nib.Nifti1Image(run_data, affine=run_img.affine)
    if img:
        return runs_fmri, runs_img
    return runs_fmri

def brain_gif_maker(run_img, run_fmri, filename, skip_frames = 1, duration = 2000, temp_dir = None):
    """
    Takes a list of fMRI images from a single run together with the fMRI data from that run and creates a GIF of the voxel activities over time.

    Args:
        run_img: the list of fMRI images loaded by nibabel
        run_fmri: the array of fMRI data of shape (# TRs, # voxels)
        filename: path to save gif
        skip_frames: only consider every other skip_frames TR when making gif (total used frames is then the total TRs available / skip_frames)
        duration: the duration of each frame in ms. Defaults to TR duration for real-time results
        temp_dir: temporary directory to save generated frames to. Will be deleted after

    Returns:

    """

    print("Creating gif at: ", filename)
    # Read the run img
    imgs = image.iter_img(run_img)
    filenames = []

    # Create directories
    if temp_dir is None:
        temp_dir = 'gif_frames'

    os.makedirs(temp_dir, exist_ok=True)

    # Plot with diverging colormap centered on mean across time
    # mean_val = np.mean(run_fmri)
    mean_val = 0
    vmax_abs = max(abs(np.max(run_fmri) - mean_val), abs(mean_val - np.min(run_fmri)))
    vmin = mean_val - vmax_abs
    vmax = mean_val + vmax_abs

    # Generate frame for each time point
    for i, img in tqdm(enumerate(imgs)):

        if i % skip_frames != 0:
            continue

        # Save each frame to a file
        fn = os.path.join(temp_dir, f"frame_{i:03d}.png")
        display = plot_glass_brain(
            img,
            cmap='seismic',
            vmin=vmin,
            vmax=vmax,
            plot_abs=False
        )
        display.savefig(fn)
        display.close()
        filenames.append(fn)

    # Load frames and save as GIF
    frames = []
    for fn in filenames:
        with Image.open(fn) as img:
            frames.append(img.copy())

    frames[0].save(filename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration,  # ms per frame
                   loop=0)
    shutil.rmtree(temp_dir)


def compute_sig_voxels(runs_fmri, test_run_numbers, log = False, crop_at = 296, alpha = 0.05):
    """
    Perform Holm-Bonferroni significance test for multiple comparisons on the signal correlations between repeated runs. Also computes the correlation between the first repeated run vs all other runs.

    Args:
        runs_fmri: dictionary mapping runs to voxels
        test_run_numbers: list of test runs
        log: flag to print significance test results
        crop_at: the max number of TRs to consider in each run in order to compare different runs to each other
        alpha: the desired type-1 error

    Returns:
        rs: dictionary mapping runs to correlations of each voxel between each run and the first repeated run (the repeated run with itself is omitted)
        ps: dictionary mapping runs to the p-value of each voxel's correlation between each run and the first repeated run (the repeated run with itself is omitted)
        sorted_voxels: dictionary mapping runs to their voxels sorted by correlation with the first repeated run
        sig_voxels: dictionary mapping runs to a binary mask that is 1 where the voxels were statistically significant and 0 otherwise
    """

    n_voxels = runs_fmri[1].shape[1]
    rs, ps, sorted_voxels, sig_voxels = {}, {}, {}, {} # correlations, p-values, voxels sorted by correlation, and binary mask of statistically significant voxels
    holm_bonferroni_thresholds = alpha / np.arange(n_voxels, 0 , -1) # https://www.statisticshowto.com/holm-bonferroni-method/
    for run_number, run_fmri in runs_fmri.items():
        if run_number == test_run_numbers[0]:
            continue
        r, p = pearsonr(run_fmri[:crop_at,:], runs_fmri[test_run_numbers[0]][:crop_at, :], axis=0)
        sorted_voxels[run_number] = np.argsort(r)
        rs[run_number] = r
        ps[run_number] = p

        # Perform Holm-Bonferroni multiple comparisons correction
        p_ranks = np.argsort(np.argsort(ps[run_number]))
        p_sorted = sorted(ps[run_number])
        sig_test = p_sorted < holm_bonferroni_thresholds
        fail_idx = np.nonzero(~sig_test)[0][0]
        sig_test[fail_idx:] = False
        sig_voxels[run_number] = sig_test[p_ranks]
        if log:
            # print(f'proportion of significant voxels {run_number}v3 {sig_voxels[run_number].sum()/n_voxels}')
            print(f'proportion of significant voxels {run_number}v3 {sig_voxels[run_number].sum()}')
    return rs, ps, sorted_voxels, sig_voxels


def load_frames(frames_dir: str, all_run_numbers: list[int], test_run_numbers: list[int], skip_frames: int = 1) -> dict[int, Iterable[Image]]:
    """
    Load frames from `frames_dir` over all specified runs.

    Args:
        frames_dir: directory where frames reside. Should contain runs as subfolders.
        all_run_numbers: list of runs to load frames for
        test_run_numbers: list of repeat runs. The last one will be assigned the frames of the first repeat.
        skip_frames: If > 1, will only load every other skip_frames frame

    Returns:
        dictionary mapping runs to array of images
    """
    runs_frames = {}
    for run_number in all_run_numbers:
        imgs = []
        run_path = join(frames_dir, f'run{run_number}')
        if run_number == test_run_numbers[1]:
            runs_frames[test_run_numbers[1]] = runs_frames[test_run_numbers[0]]
            continue
        for filename in tqdm(os.listdir(run_path)[::skip_frames]):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # Supported formats
                img_path = join(run_path, filename)
                img = Image.open(img_path).convert("RGB")  # Convert to RGB (optional)
                imgs.append(img)
        runs_frames[run_number] = np.array(imgs)
    return runs_frames

def stimulus_gif_maker(run_number, runs_frames, fps: int, target_size = (256, 256),  skip_frames: int = 1):
    """
    Generate GIF for specified run from its frames.

    Args:
        run_number:
        runs_frames:
        fps:
        target_size:
        skip_frames:

    Returns:

    """
    imgs = []
    for img in tqdm(runs_frames[run_number]):  # [::(FPS//skip_frames)*TR]
        imgs.append(
            Image.fromarray(img).resize(target_size, Image.Resampling.NEAREST))  # resize image for memory reasons
    # duration is the number of milliseconds between frames
    imgs[0].save(f"run{run_number}_stimulus.gif", save_all=True, append_images=imgs[1:],
                 duration=1 / fps * skip_frames * 1000, loop=0)

def load_CLIP_embeddings(runs_frames: dict[int, ArrayLike], all_run_numbers: list[int], test_run_numbers: list[int]) -> dict[int, ArrayLike]:
    """
    Load CLIP embeddings of frames.

    Args:
        runs_frames: dictionary mapping runs to frames
        all_run_numbers: list of specified runs
        test_run_numbers: list of test runs

    Returns:
        dictionary mapping runs to CLIP embeddings of frames
    """
    # Get embeddings from CLIP
    # Load the pretrained model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    runs_CLIP = {}
    for run_number in all_run_numbers:
        if run_number == test_run_numbers[1]:
            runs_CLIP[test_run_numbers[1]] = runs_CLIP[test_run_numbers[0]]
            break
        runs_CLIP[run_number] = []
        for img in tqdm(runs_frames[run_number]):
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():  # No need for gradients
                image_features = model.get_image_features(**inputs)

            # Normalize the embeddings (useful for similarity comparisons)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy if needed
            embedding = image_features.cpu().numpy().flatten().squeeze()
            runs_CLIP[run_number].append(embedding)
        runs_CLIP[run_number] = np.array(runs_CLIP[run_number])
    return runs_CLIP


def load_R3M_embeddings(runs_frames: dict[int, ArrayLike], all_run_numbers: list[int], test_run_numbers: list[int], transforms = None) -> dict[int, ArrayLike]:
    """
    Load R3M embeddings of frames.

    Args:
        runs_frames: dictionary mapping runs to frames
        all_run_numbers: list of specified runs
        test_run_numbers: list of test runs
        transforms: transform to apply to image

    Returns:
        dictionary mapping runs to R3M embeddings of frames
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    r3m = load_r3m("resnet50")  # resnet18, resnet34, resnet50
    r3m.eval()
    r3m.to(device)

    ## DEFINE PREPROCESSING
    if transforms is None:
        transforms = T.Compose([T.Resize(256),
                                T.CenterCrop(224),  # NOTE even if you don't crop the image, the model will crop to 224x224!
                                T.ToTensor()])  # ToTensor() divides by 255

    runs_R3M = {}
    for run_number in all_run_numbers:
        if run_number == test_run_numbers[1]:
            runs_R3M[test_run_numbers[1]] = runs_R3M[test_run_numbers[0]]
            break
        runs_R3M[run_number] = []
        for img in tqdm(runs_frames[run_number]):
            preprocessed = transforms(Image.fromarray(img, 'RGB')).reshape(-1, 3, 224, 224)
            preprocessed = preprocessed.to(device)

            # Get embedding
            with torch.no_grad():
                embedding = r3m(preprocessed * 255.0)

            # Convert to numpy if needed
            embedding = embedding.cpu().numpy()[0].flatten().squeeze()
            runs_R3M[run_number].append(embedding)
        runs_R3M[run_number] = np.array(runs_R3M[run_number])
    return runs_R3M

def load_resnet50_embeddings(runs_frames: dict[int, ArrayLike], all_run_numbers: list[int], test_run_numbers: list[int], transforms = None) -> dict[int, ArrayLike]:
    """
    Load resnet50 embeddings of frames.

    Args:
        runs_frames: dictionary mapping runs to frames
        all_run_numbers: list of specified runs
        test_run_numbers: list of test runs
        transforms: transform to apply to image

    Returns:
        dictionary mapping runs to resnet50 embeddings of frames
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ## DEFINE PREPROCESSING
    if transforms is None:
        transforms = T.Compose([T.Resize(256),
                                T.CenterCrop(224),  # NOTE even if you don't crop the image, the model will crop to 224x224!
                                T.ToTensor()])  # ToTensor() divides by 255


    resnet = resnet50(pretrained=True).to(
        device)  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    resnet.eval()
    runs_resnet50 = {}
    for run_number in all_run_numbers:
        if run_number == test_run_numbers[1]:
            runs_resnet50[test_run_numbers[1]] = runs_resnet50[test_run_numbers[0]]
            break
        runs_resnet50[run_number] = []
        for img in tqdm(runs_frames[run_number]):
            preprocessed = transforms(Image.fromarray(img, 'RGB')).reshape(-1, 3, 224, 224)
            preprocessed = preprocessed.to(device)

            # Get embedding
            with torch.no_grad():
                embedding = resnet(preprocessed * 255.0)

            # Convert to numpy if needed
            embedding = embedding.cpu().numpy()[0].flatten().squeeze()
            runs_resnet50[run_number].append(embedding)
        runs_resnet50[run_number] = np.array(runs_resnet50[run_number])
    return runs_resnet50


def load_text_embeddings(runs_narration: pd.DataFrame, all_run_numbers: list[int], test_run_numbers: list[int], tr: int,  model_name="gpt2", layer=12, bad_to_good_words: dict[str, str] = None) -> (dict[int, ArrayLike], dict[int, ArrayLike]):
    """
    Get text embeddings of narrations. Converts narrations to acceptable list of words.

    Args:
        runs_narration: DataFrame of annotations
        all_run_numbers: list of specified runs
        test_run_numbers: list of test runs
        tr: duration of each TR
        model_name: pretrained model to load from transformers package
        layer: layer to extract embeddings from

    Returns:
        - dictionary mapping runs to text embeddings of narrations
        - dictionary mapping runs to TRs that were annotated
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = model.get_input_embeddings()

    # Load annotations
    runs_narration['start_frame'] = runs_narration['start_frame'].astype(int)
    runs_narration['stop_frame'] = runs_narration['stop_frame'].astype(int)
    runs_narration['start_TR'] = (runs_narration['run_start_seconds'] // tr).astype(int)
    runs_narration['stop_TR'] = (runs_narration['run_stop_seconds'] // tr).astype(int)

    # Get text embeddings
    runs_mdl = {}
    for x in runs_narration.itertuples():
        run = x.run
        start_TR = x.start_TR
        stop_TR = x.stop_TR
        noun = x.noun
        verb = x.verb

        # Map onto acceptable words
        if bad_to_good_words and noun in bad_to_good_words:
            noun = bad_to_good_words[noun]
        if bad_to_good_words and verb in bad_to_good_words:
            verb = bad_to_good_words[verb]

        # For each TR that received this annotation, get the embedding of the noun and verb
        # This will result in these TRs sharing this same embedding
        # Sometimes, TRs will overlap in their annotations. We collect all the embeddings for a given TR in a list.
        if run not in runs_mdl:
            runs_mdl[run] = defaultdict(list)
        for tr in range(stop_TR, start_TR - 1, - 1):
            runs_mdl[run][tr] += [get_multi_string_embedding([noun, verb], tokenizer, embedding_layer)]

    # Track the location of annotated TRs
    runs_TRs = {}
    for k,v  in runs_mdl.items():
        runs_TRs[k] = np.array(sorted(list(v.keys()))).astype(int)
    runs_TRs[test_run_numbers[1]] = runs_TRs[test_run_numbers[0]]
    return runs_mdl, runs_TRs

# Taken from speechmodeltutorial
def delay_mat(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def delay_embeddings(delays, embeddings, all_run_numbers, test_run_numbers) -> dict[int, Any]:
    """
    Delay the embeddings.

    Args:
        delays:
        embeddings:
        all_run_numbers:
        test_run_numbers:

    Returns:

    """
    delayed_embeddings = {}
    for run_number in all_run_numbers:
        if run_number == test_run_numbers[1]:
            delayed_embeddings[test_run_numbers[1]] = delayed_embeddings[test_run_numbers[0]]
            break
        delayed_embeddings[run_number] = []
        delayed_embeddings[run_number] = delay_mat(embeddings[run_number], delays)
    return delayed_embeddings


def drop_fixation_TRs_from_fmri(runs_fmri, tr = 2, preproc_dropped_trs: int = 2, startup_blank_screen_secs: float = 20, ending_blank_screen_secs: float = 20, keep_TRs: dict[int, Iterable[int]] = None, log: bool = False) -> dict[int, ArrayLike]:
    # Calculate how many TRs to drop from the start and end of each run
    nTRs_drop_from_start = int((startup_blank_screen_secs / tr) - preproc_dropped_trs)  # Should be 8
    nTRs_drop_from_end = int((startup_blank_screen_secs / tr))  # Should be 10

    total_TRs_to_drop = nTRs_drop_from_start + nTRs_drop_from_end

    if log:
        print(f'{nTRs_drop_from_start} TRs dropped from the START of each run')
        print(f'{nTRs_drop_from_end} TRs dropped from the END of each run')

    runs_fmri_dropped = {}
    for run_number in runs_fmri.keys():
        init_nTR = runs_fmri[run_number].shape[0]
        runs_fmri_dropped[run_number] = runs_fmri[run_number][nTRs_drop_from_start:-nTRs_drop_from_end,
                                :]  # Drop TRs from start and end of the run

        out_nTR = runs_fmri_dropped[run_number].shape[0]
        assert init_nTR - out_nTR == total_TRs_to_drop

    # Keep only specified TRs
    if keep_TRs:
        for run_number, trs in keep_TRs.items():
            valid_trs = trs < runs_fmri_dropped[run_number].shape[0] # edge case where TR is annotated but overlaps with the fixation period
            runs_fmri_dropped[run_number] = runs_fmri_dropped[run_number][trs[valid_trs]]
    return runs_fmri_dropped

def average_embeddings_per_TR(fps, skip_frames, dim_embeddings, TR_length, runs_data, runs_TRs):
    """
    Average the embeddings in each TR. Embeddings are assumed to correspond to frames. Assumes non-video TRs have been dropped from the beginning.

    Args:
        fps: the fps of the original stimulus that the embeddings were taken from
        skip_frames: only consider every other skip_frames frame when the embeddings were actually generated (total embeddings is then the total frames available / skip_frames)
        dim_embeddings: the dimension of the feature space
        TR_length: the duration in seconds of each TR
        runs_data: dictionary mapping runs to their embeddings
        runs_TRs: dictionary mapping runs to the TRs that we want to use

    Returns:
        dictionary mapping runs to their TR-averaged embeddings
    """

    # Calculate the effective FPS
    sample_hz = fps/skip_frames
    frames_per_TR = int(sample_hz * TR_length)
    runs_avg = {}
    for run_number, run_data in tqdm(runs_data.items()):
        nTRs = len(runs_TRs[run_number]) # num annotated TRs for this run
        runs_avg[run_number] = np.zeros((nTRs, dim_embeddings))
        tr_list = sorted(runs_TRs[run_number]) # these are the annotated TR indices

        # For each specified TR in run
        for i, tr in enumerate(tr_list):
            start_frame = tr*frames_per_TR
            end_frame = (tr + 1) * frames_per_TR
            if end_frame > len(run_data): # if not enough embeddings at the end to average over
                avg_embedding = np.average(run_data[start_frame:], axis = 0)
            else:
                avg_embedding = np.average(run_data[start_frame:end_frame], axis = 0) # Average embeddings that fall inside the same TR
            runs_avg[run_number][i] = avg_embedding
    return runs_avg


def rdm_heatmap(embeddings, model_name, plot = True, return_rdm = True,
    cmap='flare', figsize=(10, 8), save_path=None):
    """
    Plots a dendrogram and a heatmap for the given embeddings.
    """
    # RDM for embeddings
    embedding_distances = pdist(embeddings, metric='correlation') #NOTE we don't reorder the embeddings here, and all the embeddings have same order, so we can directly compare these to do RSA
    embedding_rdm = squareform(embedding_distances)

    # embedding_rdm_df = pd.DataFrame(embedding_rdm)
    if plot:
        plt.figure(figsize=figsize)
        _ = sns.heatmap(embedding_rdm, cbar=True, cmap = cmap)
        plt.title(f"Representational Dissimilarity Matrix (RDM) of {model_name} embeddings")
        plt.xlabel("Stimuli")
        plt.ylabel("Stimuli")

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    if return_rdm:
        return embedding_rdm


def triangle(arr, k = 1):
    """
    Return flattened upper triangle of square matrix.

    Args:
        arr: square matrix
        k: input argument to np.triu_indices. Defaults to 1, which excludes the diagonal

    Returns:
        flattened upper triangle of input
    """

    return arr[np.triu_indices(arr.shape[0], k = k)]


def get_RDM_run_index(run, runs_shape):
    """
    Retrieve where along either axis of the RDM the run starts and ends.

    Args:
        run: run number
        runs_shape: dictionary mapping runs to any data of the shape (TR, Any)

    Returns:
        the start and end index of the run along either axis of the RDM
    """

    # Take any dictionary mapping runs to an array of shape (TR, Any)
    start_idx, end_idx = 0, 0
    for i in range(run):
        if i > 0:
            start_idx += runs_shape[i].shape[0]
        end_idx += runs_shape[i+1].shape[0]
    return start_idx, end_idx


def upper_triangle_index(i, j, n):
    """
    Given the indices into a square matrix of size n, retrieve the index into a flattened upper triangle where the diagonal is excluded.

    Args:
        i: row index
        j: column index
        n: shape of square matrix

    Returns:
        index into a flattened upper triangle where the diagonal is excluded
    """

    if i >= j:
        raise ValueError("Only valid for upper triangle with i < j")
    return (2*n - i - 1)*i//2 + (j - i - 1)


def nan_metric(u, v, metric = 'pearson'):
    """
    NaN-aware wrapper around similarity metrics.

    Args:
        u: vector 1
        v: vector 2
        metric: similarity metric

    Returns:
        similarity metric of input vectors with NaNs removed
    """

    # Mask for valid (non-NaN) positions in both vectors
    mask = ~np.isnan(u) & ~np.isnan(v)
    if not np.any(mask):
        return np.nan  # or 1.0 (max distance) to indicate no shared info

    u_masked = u[mask]
    v_masked = v[mask]

    # Normalize
    norm_u = np.linalg.norm(u_masked)
    norm_v = np.linalg.norm(v_masked)

    if norm_u == 0 or norm_v == 0:
        return np.nan  # undefined cosine similarity

    match metric:
        case 'spearman':
            return spearmanr(u_masked, v_masked).statistic
        case 'pearson':
            return pearsonr(u_masked, v_masked)[0]
        case _:
            return np.dot(u_masked, v_masked) / (norm_u * norm_v)


def permute_rdm(mat, rng = np.random.default_rng(0)):
    """
    Permute rows of a matrix

    Args:
        mat: matrix to permute
        rng: random number generator

    Returns:
        permuted matrix
    """

    n = mat.shape[0]
    # Create permutation
    perm = rng.permutation(n)
    return mat[perm, :]


def permute_rsa(rdms, rdms_full, n_perms = 20, seed = 0):
    """
    Permutation testing of RSA matrix, where the RDMs are permuted several times and the RSA is recalculated on permuted RDMs.

    Args:
        rdms: dictionary mapping models to (flattened) RDMs
        rdms_full: dictionary mapping to (square) RDMs
        n_perms: number of permutations to perform
        seed: seed for random number generator

    Returns:
        array storing each permuted RSA result
    """
    rsa_matrix_perm = np.zeros((n_perms, len(rdms), len(rdms)))
    rng = np.random.default_rng(seed)
    mdls = list(rdms.keys())
    all_run_numbers = rdms_full
    # Permute RDMs n_perms times
    for p in tqdm(range(n_perms)):
        rdms_perm = {}

        # For each model, generate permuted RDM
        for mdl in mdls:
            rdms_perm[mdl] = []
            for i in all_run_numbers:
                rdms_perm[mdl].append(triangle(permute_rdm(rdms_full[mdl][i - 1], rng)))

        # Perform RSA on permuted RDMs of all models
        mdls = list(rdms_perm.keys())
        for i in range(len(mdls)):
            for j in range(i, len(mdls)):
                if mdls[i] == mdls[j]:
                    if mdls[i] == 'fMRI' and mdls[j] == 'fMRI':
                        rsa_matrix_perm[p, i,j] = spearmanr(rdms['fMRI'][2], rdms_perm['fMRI'][5], nan_policy = 'omit').statistic
                    continue
                rep1 = np.concatenate(rdms[mdls[i]][:5])
                rep2 = np.concatenate(rdms_perm[mdls[j]][:5])
                rsa_matrix_perm[p, i, j] = spearmanr(rep1, rep2, nan_policy = 'omit').statistic
    return rsa_matrix_perm

