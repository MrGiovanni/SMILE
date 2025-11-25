import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd 

from scipy.ndimage import zoom
from skimage.transform import resize

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def erode_mask(mask, kernel_size=3):
    """
    reduce the bias caused by edges via pooling
    """
    pool = F.max_pool2d(1.0 - mask.float().unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2)
    return (1.0 - pool.squeeze(0)).bool()

def center_crop(tensor, crop_ratio=0.2):

    try:
        _, B, H, W = tensor.shape
        ch = int(H * crop_ratio)
        cw = int(W * crop_ratio)
        sh = (H - ch) // 2
        sw = (W - cw) // 2
        return tensor[:, :, sh:sh+ch, sw:sw+cw]

    except:
        B, H, W = tensor.shape
        ch = int(H * crop_ratio)
        cw = int(W * crop_ratio)
        sh = (H - ch) // 2
        sw = (W - cw) // 2
        return tensor[:, sh:sh+ch, sw:sw+cw]

def soft_dice_loss(pred_logits, target, epsilon=1e-6):
    """
    Computes soft Dice loss for multi-class prediction.
    - pred_logits: (N, C, H, W) logits
    - target: (N, H, W) class indices (int)
    """


    
    num_classes = pred_logits.shape[1]
    # Apply softmax to get probabilities
    probs = F.softmax(pred_logits, dim=1)  # (N, C, H, W)

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes)  # (N, H, W, C)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()    # (N, C, H, W)

    # Compute Dice per class
    dims = (0, 2, 3)  # sum over N, H, W
    intersection = torch.sum(probs * target_onehot, dims)
    union = torch.sum(probs + target_onehot, dims)

    dice = (2 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice.mean()
    return dice_loss







def hu_correlation(seg_pred, seg_gt, ct_image, ignore_label=0):
    """
    Return Pearson correlation between organ-wise mean HU values of GT vs Pred.
    - Standard is GT organ ids.
    - If a GT organ does not appear in Pred, set its predicted HU as 0.
    """
    labels = np.unique(seg_gt)
    labels = labels[labels != ignore_label]

    hu_gt_list, hu_pred_list = [], []

    for organ_id in labels:
        mask_gt = (seg_gt == organ_id)
        mask_pred = (seg_pred == organ_id)

        if mask_gt.any():
            hu_gt = np.mean(ct_image[mask_gt])
            hu_gt_list.append(hu_gt)

            if mask_pred.any():
                hu_pred = np.mean(ct_image[mask_pred])
            else:
                hu_pred = 0.0  # missing in prediction → set to zero
            hu_pred_list.append(hu_pred)

    if len(hu_gt_list) >= 2:
        r, _ = pearsonr(hu_gt_list, hu_pred_list)
        return float(r)
    else:
        return -1  # not enough organs for correlation


def label2rgb(image):
# Clip HU to [-1000, 1000] (common soft tissue window)
    image = np.clip(image, -1000, 1000)
    # Normalize to [0, 255]
    image = ((image + 1000) / 2000.0 * 255).astype(np.uint8)
    return image


def hu_and_size_lists(seg_pred, seg_gt, ct_image, ignore_label=0):
    """
    Return:
        size_list: [(size_gt, size_pred), ...] in voxels
        hu_list:   [(meanHU_gt, meanHU_pred), ...]
    """
    labels = np.union1d(np.unique(seg_gt), np.unique(seg_pred))
    labels = labels[labels != ignore_label]

    size_list = []
    hu_list = []

    for organ_id in labels:
        mask_gt = (seg_gt == organ_id)
        mask_pred = (seg_pred == organ_id)

        if mask_gt.any() and mask_pred.any():
            size_gt = int(mask_gt.sum())
            size_pred = int(mask_pred.sum())

            hu_gt = float(np.mean(ct_image[mask_gt]))
            hu_pred = float(np.mean(ct_image[mask_pred]))

            size_list.append((size_gt, size_pred))
            hu_list.append((hu_gt, hu_pred))

    return size_list, hu_list
    

def find_GT_BDMAP(patient_id, phase, reference_excel_path='./dataset/step3-ids.xlsx'):
    """
    Get the matched BDMAP_ID of inquery patient id and phase.
    """
    try:
        reference_df = pd.read_csv(reference_excel_path)
    except:
        reference_df = pd.read_excel(reference_excel_path)

    row = reference_df[reference_df["Patient_ID"].str.contains(patient_id)]
    return row[phase].values[0]


def resample_CT_data(CT1: np.ndarray, CT2: np.ndarray, CT3: np.ndarray, order: int = 1):
    """
    Resample three 3D CT volumes (X, Y, Z) along Z so all end up with the lowest Z.
    X and Y dimensions are kept unchanged.

    Args:
        CT1, CT2, CT3: np.ndarray with shape (X, Y, Zi)
        order: interpolation order for scipy.ndimage.zoom (0=nearest, 1=linear, 3=cubic)

    Returns:
        CT1r, CT2r, CT3r with shape (X, Y, Zmin)
    """
    assert CT1.ndim == CT2.ndim == CT3.ndim == 3, "Inputs must be 3D (X,Y,Z)."
    x1, y1, z1 = CT1.shape
    x2, y2, z2 = CT2.shape
    x3, y3, z3 = CT3.shape
    # Only resampling along Z; X/Y must match
    assert (x1, y1) == (x2, y2) == (x3, y3), "X/Y must match to resample only along Z."

    target_z = min(z1, z2, z3)

    def _resample_z(vol, z, target_z):
        if z == target_z:
            out = vol
        else:
            f = target_z / float(z)
            out = zoom(vol, zoom=(1.0, 1.0, f), order=order)
        # guard against rounding oddities
        if out.shape[2] != target_z:
            out = out[:, :, :target_z]
        return out.astype(np.float32, copy=False)

    CT1r = _resample_z(CT1, z1, target_z)
    CT2r = _resample_z(CT2, z2, target_z)
    CT3r = _resample_z(CT3, z3, target_z)
    return CT1r, CT2r, CT3r


def _minmax01(v: np.ndarray) -> np.ndarray:
    """
    minmax normalize
    """
    v = v.astype(np.float32)
    vmin, vmax = np.min(v), np.max(v)
    if vmax <= vmin + 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v - vmin) / (vmax - vmin)


def match_resolution(img1, img2):
    """Resize higher-resolution image to match the lower-resolution one."""
    if img1.shape != img2.shape:
        # Determine which is smaller (in pixel count)
        if np.prod(img1.shape) <= np.prod(img2.shape):
            target_shape = img1.shape
            img2 = resize(img2, target_shape, preserve_range=True, anti_aliasing=True)
        else:
            target_shape = img2.shape
            img1 = resize(img1, target_shape, preserve_range=True, anti_aliasing=True)
    return img1, img2



def plot_violin_plot(input_list, name_plot, save_dir):
    """
    Plots a violin plot showing the mean and distribution of input_list.

    Args:
        input_list (list or np.ndarray): The data to plot.
        name_plot (str): Title of the plot.
    """
    input_array = np.array(input_list)

    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(input_array, showmeans=True, showextrema=True, showmedians=False)

    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)

    ax.set_title(name_plot)
    ax.set_ylabel('Value')
    ax.set_xticks([1])
    ax.set_xticklabels([name_plot])

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_dir)
    plt.close()



def plot_correlation(input_list, title="Size Correlation", axis_scale="mm$^3$", save_dir="./temp_test.png"):
    """
    Plots scatter plot for size correlation and shows Pearson correlation in legend.

    Args:
        input_list (list of tuple): Each tuple is (gt, pred)
        title (str): Title of the plot
    """
    # Split into two lists
    size_gt, size_pred = zip(*input_list)
    size_gt = np.array(size_gt)
    size_pred = np.array(size_pred)

    # Compute Pearson correlation
    r, _ = pearsonr(size_gt, size_pred)

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(size_gt, size_pred, alpha=0.6, label=f"Pearson r = {r:.3f}", s=50)

    # Fit a linear regression line
    m, b = np.polyfit(size_gt, size_pred, 1)
    plt.plot(size_gt, m * size_gt + b, color="red", linestyle="--", label="Fit line")

    # Labels & legend
    plt.xlabel(f"Ground Truth Organs {axis_scale}")
    plt.ylabel(f"Translated Organs {axis_scale}")
    plt.title(title)
    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")  # Equal scaling for better visual
    plt.savefig(save_dir)
    plt.close()



label_mapping = { # noncontrast-arterial-venous-delayed sequence
            "arterial": 1,
            "venous": 2,
            "delayed": 3,
            "non-contrast": 0,
        }
index_to_phase = {v: k for k, v in label_mapping.items()}


if __name__ == '__main__':
    # Example usage
    size_list = [(10, 9), (20, 22), (15, 14), (30, 28)]
    plot_correlation(size_list)
    