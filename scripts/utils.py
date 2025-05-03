from collections.abc import Iterable

import numpy as np
from nilearn.image import resample_to_img
from numpy._typing import ArrayLike
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
from nilearn import image
import os
from os.path import join
import shutil
from tqdm import tqdm
from nilearn.plotting import  plot_glass_brain
from scipy.stats import pearsonr, zscore, spearmanr
from scipy.spatial.distance import squareform, pdist
from nilearn.masking import apply_mask, unmask
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

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


def load_fmri(fmri_dir: str, brainmask_img: nib.nifti1.Nifti1Image, run_name_template: str, all_run_numbers: list[int], ROI: str='whole', img: bool = False) -> dict[int, ArrayLike]:
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
            print(f'proportion of significant voxels {run_number}v3 {sig_voxels[run_number].sum()/n_voxels}')
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


def average_embeddings_per_TR(fps, skip_frames, dim_embeddings, TR_length, runs_data, runs_TRs):
    """
    Average the embeddings in each TR. Embeddings are assumed to correspond to frames.

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
        nTRs = len(runs_TRs[run_number])
        runs_avg[run_number] = np.zeros((nTRs, dim_embeddings))
        tr_list = sorted(runs_TRs[run_number])

        # For each specified TR in run
        for i, tr in enumerate(tr_list):
            if (tr+1)*frames_per_TR > len(run_data): # if not enough embeddings at the end to average over
                avg_embedding = np.average(run_data[tr*frames_per_TR:], axis = 0)
            else:
                avg_embedding = np.average(run_data[tr*frames_per_TR:(tr+1)*frames_per_TR], axis = 0) # Average embeddings that fall inside the same TR
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
        runs_shape: dictionary mapping runs to any data of the shape (TR, voxels)

    Returns:
        the start and end index of the run along either axis of the RDM
    """

    # Take any dictionary mapping runs to an array of shape (TR, voxels)
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

