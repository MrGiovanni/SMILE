import torch
import os
import numpy as np
import nibabel as nib
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import time
from tqdm import tqdm
import pandas as pd
import h5py
from collections import defaultdict

from utils import find_GT_BDMAP
# NOTE IMPORTANT!! 
__all_phases__ = ["non-contrast", "arterial", "venous", "delayed"]

def find_duplicate_indices(tuples_list):
    index_dict = defaultdict(list)
    for idx, tpl in enumerate(tuples_list):     # Iterate through the list once and store indices
        index_dict[tpl].append(idx)
    return {key: value for key, value in index_dict.items() if len(value) > 1}  # Extract only duplicates

def collate_fn(examples):

        valid_examples = [
                ex for ex in examples
                if ex is not None and isinstance(ex, dict) and ex.get("pixel_values") is not None
            ]
        if len(valid_examples) == 0:
            return None 
        examples = valid_examples
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_mask_values = torch.stack([example["cond_mask_values"] for example in examples])
        cond_mask_values = cond_mask_values.to(memory_format=torch.contiguous_format).float()
        gt_mask_values = torch.stack([example["mask_values"] for example in examples])
        gt_mask_values = gt_mask_values.to(memory_format=torch.contiguous_format).float()
        gt_phase_id = torch.stack([example["gt_phase_id"] for example in examples])
        gt_phase_id = gt_phase_id.to(memory_format=torch.contiguous_format).long()  
        cond_phase_id = torch.stack([example["cond_phase_id"] for example in examples])
        cond_phase_id = cond_phase_id.to(memory_format=torch.contiguous_format).long()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        cond_ids = torch.stack([example["cond_ids"] for example in examples])
        cond_pixel_values_orginal = torch.stack([example["cond_pixel_values_original"] for example in examples])
        cond_pixel_values_orginal = cond_pixel_values_orginal.to(memory_format=torch.contiguous_format).float()
        unchanged_mask = torch.stack([example["unchanged_mask"] for example in examples])
        unchanged_mask = unchanged_mask.to(memory_format=torch.contiguous_format)

        return {
            "pixel_values": pixel_values, 
            "mask_values": gt_mask_values,
            "input_ids": input_ids, 
            "cond_ids": cond_ids,   
            "cond_pixel_values": cond_pixel_values, 
            "cond_mask_values": cond_mask_values,
            "gt_phase_id": gt_phase_id,
            "cond_phase_id": cond_phase_id,
            "cond_pixel_values_original": cond_pixel_values_orginal,
            "unchanged_mask": unchanged_mask
        }


def collate_fn_inference(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_prompt = [example["input_prompt"] for example in examples]
        slice_idx = [example["slice_idx"] for example in examples]
        return {
            # "pixel_values": pixel_values, 
            "input_prompt": input_prompt,   # NOTE: different from training
            "cond_pixel_values": cond_pixel_values,
            # "gt_pixel_values": gt_pixel_values,
            "slice_idx": slice_idx
        }

def varifyh5(filename): # read the h5 file to see if the conversion is finished or not
    try:
        with h5py.File(filename, "r") as hf:   # can read successfully
            pass
        return True
    except OSError:     # transform not complete
        return False


def load_CT_slice(ct_path, slice_idx=None):
    """For AbdomenAtlasPro data: ranging from [-1000, 1000], shape of (H W D) """
    with h5py.File(ct_path, 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]

        # NOTE: take adjacent 3 slices into the 3 RGB channel
        if slice_idx is None:
            slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point
        while True:
            try:    # some slices of some CT are broken
                cond_slice = nii[:, :, slice_idx:slice_idx + 3]   
                break
            except: # if broken, randomly select until select the non-broken slice
                print(f"\033[31mBroken slice: {ct_path.split('/')[-2]}, slice {slice_idx}\033[0m")
                slice_idx = random.randint(0, z_shape - 3)

    with h5py.File(ct_path.replace(ct_path.split("/")[-3], "BDMAP_O"), 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]
        ct_slice = nii[:, :, slice_idx:slice_idx + 3]   

    with h5py.File(ct_path.replace(ct_path.split("/")[-3], "BDMAP_O").replace("ct.h5", "pred.h5"), 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]
        gt_slice = nii[:, :, slice_idx:slice_idx + 3]   

            
    # target range: [-1000, 1000] -> [0, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]

    cond_slice[cond_slice > 1000.] = 1000.    # clipping range and normalize
    cond_slice[cond_slice < -1000.] = -1000.
    cond_slice = (cond_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice, cond_slice, gt_slice  # (H W 3)[0, 1]

def load_CT_sliceh5(ct_path, slice_idx=None):
    """
    For our training data: ranging from [-1000, 1000], shape of (H W D) 

    Loading three adjacent slices as three channels, for diffusion model structure consistency.
    """
    with h5py.File(ct_path, 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]

        # NOTE: take adjacent 3 slices into the 3 RGB channel
        if slice_idx is None:
            slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point

        ct_slice = nii[:, :, slice_idx:slice_idx + 3]   
            
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.          # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    
    if ct_slice.shape[2]!= 3:
        raise ValueError

    return ct_slice  # (H W 3)[0, 1]

def load_CT_slice_from_h5py(nii, slice_idx=None):
    """
    For inference:

        AbdomenAtlasPro data: ranging from [-1000, 1000], shape of (H W D) 
    """
    
    z_shape = nii.shape[2]

    # NOTE: take adjacent 3 slices into the 3 RGB channel
    if slice_idx is None:
        slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point

    end_idx = slice_idx + 3
    if end_idx > z_shape:
        # use the last one for padding
        slices = [nii[:, :, i] for i in range(slice_idx, z_shape)]
        while len(slices) < 3:
            slices.append(slices[-1])  
        ct_slice = np.stack(slices, axis=-1)
    else:
        ct_slice = nii[:, :, slice_idx:end_idx]

    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]

def load_CT_slice_from_nfiti(ct_data, slice_idx=None):
    """For any nii data during inference: ranging from [-1000, 1000], shape of (H W D) """
    ct_slice = ct_data.dataobj[:, :, slice_idx:slice_idx + 3].copy() 
            
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]


class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)




class ReconCTDataset(Dataset):
    def __init__(self, file_paths, data_root, image_transforms=None, cond_transforms=None, tokenizer=None, resolution=512, dataset_name="dataset901"):
        """ (training only)
        Args:
            file_paths (list): List of paths to 3D CT volumes (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices.
        """
        self.data_root = data_root
        self.file_paths = file_paths
        self.image_transforms = image_transforms
        self.cond_transforms = cond_transforms  # useless
        self.tokenizer = tokenizer
        self.norm_to_zero_centered = A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=1.0,
                p=1.0
            )
    
        self.phases = __all_phases__
        self.phases_id_mapping = { # noncontrast-arterial-venous-delayed sequence
            "arterial": 1,
            "venous": 2,
            "delayed": 3,
            "non-contrast": 0,
        }
        self.ct_name_bdmapid_mapping = pd.read_csv(f"{dataset_name}/phase_translation_map.csv")
        self.resolution = resolution

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        patient_id, phase_set = self.file_paths[idx]  # random CT

        # choose input-cond pair (can be the same)
        chosen_shape = random.choice(list(phase_set.keys()))        # choose a resolution and its pairs
        chosen_sublist = phase_set[chosen_shape]                    # for all same resolution pairs of a patient, choose one
        random.shuffle(chosen_sublist)                              # shuffle the same resolution pairs
        chosen_phases_idx = random.choices(chosen_sublist, k=2)     # 1st as input, 2nd as cond
        
        input_ct_phase = self.phases[chosen_phases_idx[0]]
        cond_ct_phase = self.phases[chosen_phases_idx[1]] 
        input_phase_file_name = f"{patient_id}_{input_ct_phase}"
        cond_phase_file_name = f"{patient_id}_{cond_ct_phase}"

        
        # bdmap mapping
        input_h5_path = self.ct_name_bdmapid_mapping[self.ct_name_bdmapid_mapping['original_name']==input_phase_file_name]['bdmap_id'].values[0]
        cond_h5_path = self.ct_name_bdmapid_mapping[self.ct_name_bdmapid_mapping['original_name']==cond_phase_file_name]['bdmap_id'].values[0]

        # print(f"Loading patient no. {patient_id}, bdmap id. {input_h5_path} & {cond_h5_path}, chosen phase {input_ct_phase} & {cond_ct_phase}")
        input_ct_path = os.path.join(self.data_root, input_phase_file_name, "ct.h5")
        cond_ct_path = os.path.join(self.data_root, cond_phase_file_name, "ct.h5")
        input_gt_path = os.path.join(self.data_root, input_phase_file_name, "gt.h5")
        cond_gt_path = os.path.join(self.data_root, cond_phase_file_name, "gt.h5")

        with h5py.File(input_ct_path, 'r') as hf:
            nii = hf['image']
            input_z_shape = nii.shape[2]

        with h5py.File(cond_ct_path, 'r') as hf:
            nii = hf['image']
            cond_z_shape = nii.shape[2]

        
        # load CT slice
        try:
            rel_pos = random.uniform(0.05, 0.95)
            input_slice_idx = int(rel_pos * (input_z_shape - 1))
            cond_slice_idx  = int(rel_pos * (cond_z_shape  - 1))

            # then load the corresponding slice
            input_ct_slice = load_CT_sliceh5(input_ct_path, slice_idx=input_slice_idx)
            input_gt_slice = load_CT_sliceh5(input_gt_path, slice_idx=input_slice_idx)
            cond_ct_slice  = load_CT_sliceh5(cond_ct_path,  slice_idx=cond_slice_idx)
            cond_gt_slice  = load_CT_sliceh5(cond_gt_path,  slice_idx=cond_slice_idx)
        
        except:
            # add protection for broken CT
            print(f"Bad CT slice {input_ct_path.split('/')[-2]}, skipping ...")
            return None        

        cond_ct_orginal = cond_ct_slice.copy()
        cond_ct_slice_hu = cond_ct_slice*2000. - 1000.
        # 1 where cond ct is unchanged (air or bone), 0 where cond ct is changed (soft tissue)
        cond_ct_unchanged_mask = ((cond_ct_slice_hu < -800) | (cond_ct_slice_hu > 800))

        # add resolution save check
        if cond_ct_slice.shape[1] != self.resolution:
            target_size = (self.resolution, self.resolution)
            input_ct_slice = cv2.resize(input_ct_slice, target_size, interpolation=cv2.INTER_LINEAR)
            cond_ct_slice  = cv2.resize(cond_ct_slice,  target_size, interpolation=cv2.INTER_LINEAR)
            input_gt_slice = cv2.resize(input_gt_slice.astype('uint8'), target_size, interpolation=cv2.INTER_NEAREST)
            cond_gt_slice  = cv2.resize(cond_gt_slice.astype('uint8'),  target_size, interpolation=cv2.INTER_NEAREST)

        # cond transform safecheck (possibility < 0.1%)
        try:
            ct_cond_transformed = self.image_transforms(
                image=input_ct_slice, 
                cond=cond_ct_slice, 
                mask=cond_gt_slice
            )

            ct_input_transformed = self.image_transforms(
                image=input_ct_slice, 
                cond=cond_ct_slice, 
                mask=input_gt_slice
            )

            ct_slice = ct_cond_transformed["image"]  
            gt_slice = ct_input_transformed["mask"]
            cond_ct_slice = ct_cond_transformed["cond"]  
            cond_gt_slice = ct_cond_transformed["mask"]

            ct_slice = HWCarrayToCHWtensor(p=1.)(
                image=self.norm_to_zero_centered(
                    image=ct_slice)["image"]
                    )["image"]   # array to tensor
            cond_ct_slice = HWCarrayToCHWtensor(p=1.)(
                image=self.norm_to_zero_centered(
                    image=cond_ct_slice)["image"]
                    )["image"] # array to tensor
            gt_slice = HWCarrayToCHWtensor(p=1.)(image=gt_slice)["image"]
            cond_gt_slice = HWCarrayToCHWtensor(p=1.)(image=cond_gt_slice)["image"]

            # text embedding
            text_prompt = f"A {input_ct_phase} CT slice."
            source_prompt = f"A {cond_ct_phase} CT slice."

            # the cond original value only need resize transform
            # add resolution savecheck
            if cond_ct_orginal.shape[1] != self.resolution:
                cond_ct_orginal = cv2.resize(cond_ct_orginal, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
                cond_ct_unchanged_mask = cv2.resize(cond_ct_unchanged_mask.astype("uint8"), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
            cond_ct_orginal = HWCarrayToCHWtensor(p=1.)(image=cond_ct_orginal)["image"]
            cond_ct_unchanged_mask = HWCarrayToCHWtensor(p=1.)(image=cond_ct_unchanged_mask)["image"]
        
        except:
            print(f"Bad Shape: {input_ct_slice.shape, cond_ct_slice.shape, input_gt_slice.shape, cond_gt_slice.shape}")
            return None


        example = dict()
        example["mask_values"] = gt_slice#translation target mask
        example["pixel_values"] = ct_slice#translation target pixel value
        example["cond_pixel_values"] = cond_ct_slice#cond pixel value
        example["cond_mask_values"] = cond_gt_slice#cond ct mask
        example["input_ids"] = self.tokenize_caption(text_prompt)#tokenized prompt
        example["cond_ids"] = self.tokenize_caption(source_prompt)#source promt for cycle diffusion
        example["gt_phase_id"] = torch.tensor(self.phases_id_mapping[input_ct_phase.lower()]).long()# translation target phase
        example["cond_phase_id"] = torch.tensor(self.phases_id_mapping[cond_ct_phase.lower()]).long()# original phase
        example["cond_pixel_values_original"] = cond_ct_orginal #for pixel-wise loss ONLY
        example["unchanged_mask"] = cond_ct_unchanged_mask
        return example  # Shape: (C, H, W)
    
    def tokenize_caption(self, text_prompt, is_train=True):
        captions = text_prompt
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids 
    

class CTDatasetInference(Dataset):    # for a single CT volume
    """
    General inference dataloader, for .nii.gz form data (recommanded)
    
    """
    def __init__(self, file_path, image_transforms=None, cond_transforms=None):
        """ (inference on CT volume only)
        Args:
            file_path (string): The CT volume to inference (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices. 
        """

        # read CT volume data
        self.file_path = file_path
        self.bdmap_id = self.file_path.split("/")[-2]
        self.ct_volume_nii = nib.load(self.file_path)
        self.ct_xyz_shape = self.ct_volume_nii.shape   # (H W D)
        self.ct_z_shape = self.ct_xyz_shape[2]
        
        # normalization
        self.norm_to_zero_centered = A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=1.0,
                p=1.0
            )

        ### Deprecated
        self.image_transforms = image_transforms


    def __len__(self):
        return self.ct_z_shape-3 + 1   # 3 adjacent clices as input unit

    def __getitem__(self, slice_idx): # slice_idx will always in order by setting `shuffle=False`
        cond_ct_slice_raw = load_CT_slice_from_nfiti(self.ct_volume_nii, slice_idx)     # [0, 1]
        cond_ct_slice = self.image_transforms(image=cond_ct_slice_raw)["image"]

        if cond_ct_slice.shape[2] !=3:# border protection
            return None

        cond_ct_slice = HWCarrayToCHWtensor(p=1.)(
            image=self.norm_to_zero_centered(
                image=cond_ct_slice)["image"]
                )["image"] # array to tensor    [0, 1] -> ~[-1, 1]
        
        text_prompt = f""   # default
        example = dict()
        example["cond_pixel_values"] = cond_ct_slice
        example["input_prompt"] = text_prompt
        example["slice_idx"] = slice_idx    # haha.

        return example  # Shape: (C, H, W)



class CTDatasetInferenceH5(Dataset):    # for a single CT volume
    """
    The inference container for .H5 dataset form
    
    """
    def __init__(self, file_path, image_transforms=None, cond_transforms=None):
        """ (inference on CT volume only)
        Args:
            file_path (string): The CT volume to inference (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices. 
        """
        # read CT volume data
        self.file_path = file_path
        self.bdmap_id = self.file_path.split("/")[-2]
        self.ct_volume_nii = h5py.File(self.file_path, 'r')['image']  # directly read the 'image' dataset
        # self.ct_volume_data = self.ct_volume_nii.get_fdata()
        self.ct_xyz_shape = self.ct_volume_nii.shape   # (H W D)
        self.ct_z_shape = self.ct_xyz_shape[2]
        
        # normalization
        self.norm_to_zero_centered = A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=1.0,
                p=1.0
            )

        ### Deprecated
        self.image_transforms = image_transforms
        self.phases = __all_phases__

    def __len__(self):
        return self.ct_z_shape   # 3 adjacent clices as input unit

    def __getitem__(self, slice_idx, phase='arterial'): # slice_idx will always in order by setting `shuffle=False`
        cond_ct_slice_raw = load_CT_slice_from_h5py(self.ct_volume_nii, slice_idx)     # [0, 1]
        cond_ct_slice = self.image_transforms(image=cond_ct_slice_raw)["image"]

        cond_ct_slice = HWCarrayToCHWtensor(p=1.)(
            image=self.norm_to_zero_centered(
                image=cond_ct_slice)["image"]
                )["image"] # array to tensor    [0, 1] -> ~[-1, 1]
        
        text_prompt = f""   # default

        example = dict()
        example["cond_pixel_values"] = cond_ct_slice
        example["input_prompt"] = text_prompt
        example["slice_idx"] = slice_idx    # haha.

        return example  # Shape: (C, H, W)





if __name__ == "__main__":
    train_data_dir = "/mnt/T9/AbdomenAtlas/image_mask_h5"
    paths = sorted([entry.path for entry in os.scandir(train_data_dir)
                            if entry.name.startswith("BDMAP_A") or entry.name.startswith("BDMAP_V")])
    paths = [entry.path.replace("ct.h5", "") for path in  paths
                                            for entry in os.scandir(path) if entry.name == "ct.h5"]
    print(len(paths), "CT scans found!")


    train_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop((512, 512), scale=(0.75, 1.0), ratio=(1., 1.), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        HWCarrayToCHWtensor(p=1.),
    ])

    train_dataset = CTDataset(paths, transform=train_transforms)

    def collate_fn(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = torch.stack([example for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values}#, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    for batch in tqdm(train_dataloader):
        batch = batch["pixel_values"]