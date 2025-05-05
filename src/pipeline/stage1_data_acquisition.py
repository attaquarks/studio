# ========== Stage 1: Data Acquisition and Preprocessing ==========
import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

# --- Configuration ---
IMG_SIZE = 224
NUM_SLICES_PER_SCAN = 64 # Or implement dynamic padding/selection
MRI_TYPE_SUFFIX = 't1ce.nii.gz' # Example: Adjust based on your filename convention
# IMPORTANT: Calculate these values from your training dataset
DATASET_MEAN = [0.5] # Placeholder mean for normalization (per channel)
DATASET_STD = [0.5]  # Placeholder std dev for normalization (per channel)
BATCH_SIZE = 4
NUM_WORKERS = 2 # For DataLoader

# --- Dataset Class ---
class MRIDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing 3D MRI scans into 2D slices.
    Handles:
    - Loading NIfTI files (.nii, .nii.gz).
    - Extracting axial slices.
    - Selecting/padding slices to a fixed number.
    - Applying normalization and resizing.
    - Optional augmentation.
    """
    def __init__(self, file_paths, labels, mri_type_suffix=MRI_TYPE_SUFFIX,
                 num_slices=NUM_SLICES_PER_SCAN, img_size=IMG_SIZE,
                 mean=DATASET_MEAN, std=DATASET_STD, augment=False):
        """
        Args:
            file_paths (list): List of paths to 3D MRI volume files or directories containing them.
            labels (list): List of corresponding labels (str for reports, dict for VQA).
            mri_type_suffix (str): Suffix to identify the MRI file if path is a directory.
                                    If file_paths directly point to .nii.gz, this might not be needed.
            num_slices (int): Target number of axial slices per scan.
            img_size (int): Target dimension (height and width) for resizing slices.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            augment (bool): Whether to apply data augmentation.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.mri_type_suffix = mri_type_suffix
        self.num_slices = num_slices
        self.img_size = img_size
        self.augment = augment

        # Basic Transformations (applied always)
        # NOTE: ToTensor converts numpy array HWC [0, 255] or PIL Image to FloatTensor CHW [0, 1]
        # If input is already numpy float32 [0,1], ToTensor just changes HWC -> CHW
        # Our _load_nifti prepares float32 numpy slices, already scaled [0, 1].
        self.base_transform_list = [
            transforms.ToTensor(), # HWC (or HW if no channel dim yet) -> CHW ; expects [0,1] float or [0,255] uint8
            # Add channel dimension if ToTensor didn't (e.g., input was HW numpy) - should handle this in getitem
            # transforms.Lambda(lambda x: x if x.shape[0] == 1 else x.unsqueeze(0)), # redundant if ToTensor works correctly on HW
            transforms.Resize((self.img_size, self.img_size), antialias=True),
             # Normalize requires input to be Tensor (CHW)
            transforms.Normalize(mean=mean, std=std),
        ]

        self.base_transform = transforms.Compose(self.base_transform_list)


        # Augmentation Transformations (applied only if self.augment is True)
        # Applied AFTER base transforms (esp. ToTensor)
        self.augment_transform = transforms.Compose([
            # Apply transforms expecting Tensor (CHW) input
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Use carefully for medical images
            # Add other relevant augmentations if needed
        ])

    def __len__(self):
        return len(self.file_paths)

    def _load_nifti(self, file_path):
        """Loads a NIfTI file and returns the data array."""
        try:
            # Check if path is a directory (e.g., BraTS structure) or a file
            if os.path.isdir(file_path):
                 scan_name = os.path.basename(file_path)
                 mri_scan_path = os.path.join(file_path, f'{scan_name}_{self.mri_type_suffix}')
                 if not os.path.exists(mri_scan_path):
                      # Fallback if specific suffix not found, maybe the directory path *is* the filename base
                      mri_scan_path_alt = file_path + '_' + self.mri_type_suffix
                      if os.path.exists(mri_scan_path_alt):
                           mri_scan_path = mri_scan_path_alt
                      else:
                           # Try common extensions if suffix logic fails
                           for ext in ['.nii.gz', '.nii']:
                                potential_path = os.path.join(file_path, scan_name + ext)
                                if os.path.exists(potential_path):
                                    mri_scan_path = potential_path
                                    warnings.warn(f"Used fallback extension '{ext}' for {file_path}")
                                    break
            else:
                 # Assumes file_path is the direct path to the .nii or .nii.gz file
                 mri_scan_path = file_path

            if not os.path.exists(mri_scan_path):
                 warnings.warn(f"Final attempt failed. File not found: {mri_scan_path}. Skipping.")
                 return None

            mri_img = nib.load(mri_scan_path)
            # Ensure data is oriented anatomically if possible (e.g., RAS+)
            # mri_img = nib.as_closest_canonical(mri_img) # Optional: standardize orientation
            mri_data = mri_img.get_fdata(dtype=np.float32) # Load as float32

            # Clip extreme values and scale to [0, 1] before ToTensor
            # More robust scaling: percentile clipping + min-max
            p_low, p_high = np.percentile(mri_data, [1, 99]) # Clip 1% low, 1% high
            mri_data = np.clip(mri_data, p_low, p_high)

            min_val = np.min(mri_data)
            max_val = np.max(mri_data)
            if max_val > min_val:
                mri_data = (mri_data - min_val) / (max_val - min_val)
            else:
                mri_data = np.zeros_like(mri_data) # Handle constant volume case


            # Expected shape (X, Y, Z) - We want axial slices (usually Z)
            # Permute if necessary, e.g., if Z is not the last dimension
            # Example: if shape is (Z, X, Y), permute to (X, Y, Z) -> (1, 2, 0)
            # Assuming axial slices are along the last dimension (index 2) - COMMON for NIfTI
            # Transpose to (Z, X, Y) for easier slicing along Z
            # Double-check your data's dimension order!
            if mri_data.shape[-1] < mri_data.shape[0] and mri_data.shape[-1] < mri_data.shape[1]:
                # Likely Z is the last dimension, transpose to (Z, H, W)
                mri_data = mri_data.transpose(2, 0, 1) # Now shape is (NumSlices, Height, Width)
            elif mri_data.shape[0] < mri_data.shape[1] and mri_data.shape[0] < mri_data.shape[2]:
                # Likely Z is the first dimension (Z, H, W), keep as is
                 pass # Already (NumSlices, Height, Width)
            else:
                 # Default NIfTI is often (H, W, Z). Try this assumption.
                 warnings.warn(f"Ambiguous NIfTI dimension order for {mri_scan_path}. Assuming (H, W, Z) -> Transposing to (Z, H, W). Check this!")
                 mri_data = mri_data.transpose(2, 0, 1) # Transpose last dim to first

            return mri_data
        except Exception as e:
            warnings.warn(f"Error loading NIfTI file {file_path} (resolved to {mri_scan_path if 'mri_scan_path' in locals() else 'unknown'}): {e}")
            return None

    def _select_and_pad_slices(self, mri_data):
        """Selects a subset of slices or pads to reach self.num_slices."""
        current_slices = mri_data.shape[0]
        target_slices = self.num_slices

        if current_slices == 0:
             # Handle case where loading failed or scan has 0 slices
             # Use known dimensions if available, otherwise fallback to img_size
             h = mri_data.shape[1] if mri_data.ndim > 1 and mri_data.shape[1] > 0 else self.img_size
             w = mri_data.shape[2] if mri_data.ndim > 2 and mri_data.shape[2] > 0 else self.img_size
             warnings.warn(f"Input data has 0 slices. Padding with zeros of shape ({target_slices}, {h}, {w}).")
             return np.zeros((target_slices, h, w), dtype=np.float32)


        if current_slices == target_slices:
            return mri_data
        elif current_slices > target_slices:
            # Select N slices (e.g., uniformly spaced or middle N)
            # Middle N slices:
            center = current_slices // 2
            start = max(0, center - target_slices // 2)
            # Ensure we don't exceed bounds with the end index
            end = min(start + target_slices, current_slices)
            # Recalculate start if end hit the boundary
            start = max(0, end - target_slices)
            selected_slices = mri_data[start:end, :, :]
            # Final check if somehow the slice count is still wrong (shouldn't happen with this logic)
            if selected_slices.shape[0] != target_slices:
                 warnings.warn(f"Slice selection logic error: got {selected_slices.shape[0]} slices, expected {target_slices}. Using fallback linspace.")
                 indices = np.linspace(0, current_slices - 1, target_slices, dtype=int)
                 selected_slices = mri_data[indices, :, :]
            return selected_slices
        else: # current_slices < target_slices
            # Pad with zeros (background)
            padding_size = target_slices - current_slices
            pad_before = padding_size // 2
            pad_after = padding_size - pad_before
            # np.pad format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            padding = [(pad_before, pad_after), (0, 0), (0, 0)]
            padded_slices = np.pad(mri_data, padding, mode='constant', constant_values=0)
            return padded_slices

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx] # Report string or dict {'question': ..., 'answer': ...}

        mri_data_3d = self._load_nifti(file_path)

        if mri_data_3d is None or mri_data_3d.size == 0:
            # Return dummy data if loading failed or yielded empty array
            warnings.warn(f"Returning dummy data for index {idx} due to loading error or empty data.")
            # Create plausible dummy slice data (e.g., noise or zeros)
            # Shape: (num_slices, H, W) - base_transform will handle channel & resize
            # Use img_size as H, W for the dummy slice
            slices_data = np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)

        else:
            slices_data = self._select_and_pad_slices(mri_data_3d) # Shape: (num_slices, H, W)


        # Apply transforms to each slice
        processed_slices_list = []
        for i in range(self.num_slices):
            # Get slice - Shape (H, W) numpy array, assumed scaled [0, 1] by _load_nifti
            slice_img = slices_data[i, :, :]

            # Apply base transforms (ToTensor, Resize, Normalize)
            # ToTensor expects HWC or HW. Our slice_img is HW float32 [0,1].
            # It should convert HW -> 1HW (add channel dim)
            try:
                transformed_slice = self.base_transform(slice_img) # Output shape (C, H_resized, W_resized)

                # Apply augmentations if enabled
                if self.augment:
                    transformed_slice = self.augment_transform(transformed_slice)

                processed_slices_list.append(transformed_slice)
            except Exception as e:
                 warnings.warn(f"Error transforming slice {i} for index {idx}: {e}. Skipping slice.")
                 # Append a dummy tensor of the correct target size if a slice fails transform
                 # Create a dummy HWC tensor first (compatible with ToTensor)
                 dummy_slice_hw = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                 dummy_transformed = self.base_transform(dummy_slice_hw) # Apply transforms to get correct channels/size/norm
                 processed_slices_list.append(dummy_transformed)


        # If all slices failed (unlikely but possible), processed_slices_list might be empty
        if not processed_slices_list:
             warnings.warn(f"All slices failed transformation for index {idx}. Returning batch of dummy slices.")
             dummy_slice_hw = np.zeros((self.img_size, self.img_size), dtype=np.float32)
             dummy_transformed = self.base_transform(dummy_slice_hw)
             # Ensure 3 channels if needed by repeating the dummy transformed slice
             if dummy_transformed.shape[0] == 1:
                  dummy_transformed = dummy_transformed.repeat(3, 1, 1)
             processed_slices = torch.stack([dummy_transformed] * self.num_slices) # (N, C, H, W)
        else:
             # Stack successfully processed slices
             processed_slices = torch.stack(processed_slices_list) # Shape: (num_slices, C, H_resized, W_resized)


        # Ensure correct number of channels (e.g., repeat grayscale for models expecting 3 channels)
        # Check the number of channels after base_transform and stacking
        num_channels = processed_slices.shape[1]
        # Check if model expects 3 channels (common for ImageNet pre-trained models)
        # This check might need refinement based on the specific vision encoder used later
        # Example: Check if the first transform is ToTensor and output C=1
        if num_channels == 1:
             processed_slices = processed_slices.repeat(1, 3, 1, 1) # Repeat channel dim 3 times -> (N, 3, H, W)

        # --- Prepare Output ---
        item = {'pixel_values': processed_slices} # Key often used by vision models

        # Add labels (tokenization will happen later, likely in the training step)
        if isinstance(label, dict): # VQA Task
            # Ensure keys are consistent
            if 'question' in label and 'answer' in label:
                item['question'] = label['question']
                item['answer'] = label['answer'] # Ground truth answer
            else:
                warnings.warn(f"Label for index {idx} is a dict but missing 'question' or 'answer'. Label: {label}")
                # Provide default/empty values to avoid errors downstream
                item['question'] = ""
                item['answer'] = ""
        elif isinstance(label, str): # Report Generation Task
            item['report'] = label # Ground truth report
        else:
            warnings.warn(f"Unexpected label type for index {idx}. Type: {type(label)}, Label: {label}")
            # Handle unexpected label type gracefully if needed
            item['report'] = "" # Or set appropriate default


        return item

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    # Placeholder data paths and labels (USE ABSOLUTE OR CORRECT RELATIVE PATHS)
    # Create dummy NIfTI files if they don't exist for testing
    dummy_dir = './dummy_mri_data'
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_file_paths = []
    for i in range(1, 3):
        scan_name = f'scan{i}'
        scan_dir = os.path.join(dummy_dir, scan_name)
        os.makedirs(scan_dir, exist_ok=True)
        file_name = f'{scan_name}_{MRI_TYPE_SUFFIX}'
        file_path = os.path.join(scan_dir, file_name)
        if not os.path.exists(file_path):
             # Create a simple dummy NIfTI file (e.g., 64x64x30 with varying intensity)
             dummy_volume = np.random.rand(64, 64, 30).astype(np.float32) * 255
             # Add some structure
             dummy_volume[10:20, 10:20, 5:15] = 350
             dummy_volume[40:50, 40:50, 20:25] = 50
             nifti_img = nib.Nifti1Image(dummy_volume, affine=np.eye(4))
             nib.save(nifti_img, file_path)
             print(f"Created dummy file: {file_path}")
        # Add the DIRECTORY path to the list, as the dataset class handles the suffix
        dummy_file_paths.append(scan_dir) # Use directory path

    # Add a path that points directly to a file (if needed for testing)
    # file_path_direct = os.path.join(dummy_dir, 'scan1', f'scan1_{MRI_TYPE_SUFFIX}')
    # dummy_file_paths.append(file_path_direct) # Use direct file path

    # Add a non-existent path to test error handling
    dummy_file_paths.append('./non_existent_dir')


    dummy_labels = [
        {'question': 'Is there a tumor?', 'answer': 'Yes, a large enhancing tumor is present.'}, # VQA example
        'Findings: Examination reveals a well-defined lesion in the left temporal lobe. Impression: Likely glioma.', # Report gen example
        'Report for non-existent scan.' # Label for the non-existent path
    ] * (len(dummy_file_paths) // 3 + 1) # Repeat labels to match paths
    dummy_labels = dummy_labels[:len(dummy_file_paths)] # Trim labels to size


    print("--- Stage 1 Example ---")
    # Create Dataset instances
    try:
        print("Creating Training Dataset...")
        train_dataset = MRIDataset(dummy_file_paths, dummy_labels, augment=True)
        print("\nCreating Validation Dataset...")
        val_dataset = MRIDataset(dummy_file_paths, dummy_labels, augment=False) # No augmentation for validation

        # Test getting a single item
        print("\nTesting single item retrieval (first valid item)...")
        # Find first valid path index
        first_valid_idx = -1
        for i, p in enumerate(dummy_file_paths):
            # Check if directory path exists OR if it's a direct file path that exists
            full_path_check = os.path.join(p, os.path.basename(p) + '_' + MRI_TYPE_SUFFIX) if os.path.isdir(p) else p
            if os.path.exists(p) or os.path.exists(full_path_check):
                 # More thorough check for directory case
                 if os.path.isdir(p):
                      scan_name = os.path.basename(p)
                      mri_scan_path = os.path.join(p, f'{scan_name}_{MRI_TYPE_SUFFIX}')
                      if os.path.exists(mri_scan_path):
                            first_valid_idx = i
                            break
                 elif os.path.isfile(p): # Direct file path case
                      first_valid_idx = i
                      break

        if first_valid_idx != -1:
            single_item = train_dataset[first_valid_idx]
            print(f"Single item keys: {single_item.keys()}")
            print(f"Single item pixel values shape: {single_item['pixel_values'].shape}") # Should be (NumSlices, Channels, ImgSize, ImgSize)
            print(f"Pixel values dtype: {single_item['pixel_values'].dtype}")
            print(f"Pixel values min/max: {single_item['pixel_values'].min():.2f} / {single_item['pixel_values'].max():.2f}")
            if 'question' in single_item:
                print(f"Single item question: {single_item['question']}")
                print(f"Single item answer: {single_item['answer']}")
            if 'report' in single_item:
                print(f"Single item report: {single_item['report']}")
        else:
             print("No valid dummy paths found to test single item retrieval.")

        print("\nTesting single item retrieval (non-existent item)...")
        if './non_existent_dir' in dummy_file_paths:
            non_existent_idx = dummy_file_paths.index('./non_existent_dir')
            single_item_error = train_dataset[non_existent_idx]
            print(f"Error item keys: {single_item_error.keys()}")
            print(f"Error item pixel values shape: {single_item_error['pixel_values'].shape}") # Should be dummy shape
            print(f"Error item pixel values min/max: {single_item_error['pixel_values'].min():.2f} / {single_item_error['pixel_values'].max():.2f}")
        else:
            print("Non-existent path not found in dummy paths list.")


        # Create DataLoaders
        print("\nTesting DataLoader...")
        # Use num_workers=0 for easier debugging, set > 0 for performance
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Get a sample batch
        if len(train_dataset) > 0 and first_valid_idx != -1 : # Ensure there's data to load
             sample_batch = next(iter(train_loader))
             print(f"\nSample batch loaded. Keys: {sample_batch.keys()}")
             print(f"Batch pixel values shape: {sample_batch['pixel_values'].shape}") # Should be (BatchSize, NumSlices, Channels, ImgSize, ImgSize)
             if 'question' in sample_batch:
                 print(f"Batch sample question: {sample_batch['question'][0]}")
                 print(f"Batch sample answer: {sample_batch['answer'][0]}")
             if 'report' in sample_batch:
                 print(f"Batch sample report: {sample_batch['report'][0]}")
        else:
             print("Skipping DataLoader test due to empty or invalid dataset.")

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure necessary libraries (torch, torchvision, numpy, nibabel) are installed.")
    except Exception as e:
        print(f"\nError during dataset/dataloader test: {e}")
        import traceback
        traceback.print_exc()


    print("\nStage 1: Data loading and preprocessing setup complete.\n")

# Optional: Clean up dummy files (uncomment to use)
# import shutil
# try:
#     if os.path.exists(dummy_dir):
#          shutil.rmtree(dummy_dir)
#          print(f"Cleaned up dummy data directory: {dummy_dir}")
# except Exception as e:
#      print(f"Error cleaning up dummy data directory: {e}")
