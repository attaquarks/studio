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
            file_paths (list): List of paths to 3D MRI volume files.
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
        # We assume _load_nifti gives float32 numpy, maybe not in [0,1] initially.
        # It's safer to manually scale before ToTensor if necessary, or ensure nibabel loads into a range
        # that ToTensor handles correctly. For now, assuming ToTensor scales correctly or data is pre-scaled.
        self.base_transform_list = [
            transforms.ToTensor(), # HWC (or HW if no channel dim yet) -> CHW ; Scales to [0, 1] if input uint8
            # Add channel dimension if ToTensor didn't (e.g., input was HW numpy)
            transforms.Lambda(lambda x: x if x.shape[0] == 1 else x.unsqueeze(0)),
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
            else:
                 # Assumes file_path is the direct path to the .nii or .nii.gz file
                 mri_scan_path = file_path

            if not os.path.exists(mri_scan_path):
                 warnings.warn(f"File not found: {mri_scan_path}. Skipping.")
                 return None

            mri_img = nib.load(mri_scan_path)
            # Ensure data is oriented anatomically if possible (e.g., RAS+)
            # mri_img = nib.as_closest_canonical(mri_img) # Optional: standardize orientation
            mri_data = mri_img.get_fdata(dtype=np.float32) # Load as float32

            # Clip extreme values and scale to ~[0, 1] before ToTensor if needed
            # This depends heavily on the dataset's intensity distribution
            # Example: Simple min-max scaling per volume (might not be robust)
            min_val = np.min(mri_data)
            max_val = np.max(mri_data)
            if max_val > min_val:
                mri_data = (mri_data - min_val) / (max_val - min_val)
            else:
                mri_data = np.zeros_like(mri_data) # Handle constant volume case


            # Expected shape (X, Y, Z) - We want axial slices (usually Z)
            # Permute if necessary, e.g., if Z is not the last dimension
            # Example: if shape is (Z, X, Y), permute to (X, Y, Z) -> (1, 2, 0)
            # Assuming axial slices are along the last dimension (index 2) - COMMON
            # Transpose to (Z, X, Y) for easier slicing along Z
            # Double-check your data's dimension order!
            if mri_data.shape[-1] < mri_data.shape[0] and mri_data.shape[-1] < mri_data.shape[1]:
                # Likely Z is the last dimension, transpose to (Z, H, W)
                mri_data = mri_data.transpose(2, 0, 1) # Now shape is (NumSlices, Height, Width)
            elif mri_data.shape[0] < mri_data.shape[1] and mri_data.shape[0] < mri_data.shape[2]:
                # Likely Z is the first dimension (Z, H, W), keep as is
                 pass # Already (NumSlices, Height, Width)
            else:
                 # Ambiguous or needs specific check. Assuming (H, W, Z) is default NIfTI.
                 warnings.warn(f"Ambiguous NIfTI dimension order for {file_path}. Assuming (H, W, Z) -> Transposing to (Z, H, W). Check this!")
                 mri_data = mri_data.transpose(2, 0, 1)

            return mri_data
        except Exception as e:
            warnings.warn(f"Error loading NIfTI file {file_path}: {e}")
            return None

    def _select_and_pad_slices(self, mri_data):
        """Selects a subset of slices or pads to reach self.num_slices."""
        current_slices = mri_data.shape[0]
        target_slices = self.num_slices

        if current_slices == 0:
             # Handle case where loading failed or scan has 0 slices
             # Use img_size for H, W dimensions of the empty slice
             return np.zeros((target_slices, mri_data.shape[1] if mri_data.ndim > 1 else self.img_size, mri_data.shape[2] if mri_data.ndim > 2 else self.img_size), dtype=np.float32)


        if current_slices == target_slices:
            return mri_data
        elif current_slices > target_slices:
            # Select N slices (e.g., uniformly spaced or middle N)
            # Uniformly spaced:
            indices = np.linspace(0, current_slices - 1, target_slices, dtype=int)
            selected_slices = mri_data[indices, :, :]
            # Middle N slices (alternative):
            # center = current_slices // 2
            # start = max(0, center - target_slices // 2)
            # end = start + target_slices
            # Adjust if selection goes out of bounds (e.g., odd target_slices)
            # end = min(current_slices, end)
            # start = end - target_slices
            # selected_slices = mri_data[start:end, :, :]
            return selected_slices
        else: # current_slices < target_slices
            # Pad with zeros (or edge values)
            padding_size = target_slices - current_slices
            pad_before = padding_size // 2
            pad_after = padding_size - pad_before
            # np.pad format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            # Ensure padding dimensions match mri_data dimensions (Z, H, W)
            padding = [(pad_before, pad_after), (0, 0), (0, 0)]
            padded_slices = np.pad(mri_data, padding, mode='constant', constant_values=0) # Pad with 0 (background)
            return padded_slices

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx] # Report string or dict {'question': ..., 'answer': ...}

        mri_data_3d = self._load_nifti(file_path)

        if mri_data_3d is None:
            # Return dummy data if loading failed
            warnings.warn(f"Returning dummy data for index {idx} due to loading error.")
            # Create plausible dummy slice data (e.g., noise or zeros)
            # Shape: (num_slices, H, W) - base_transform will handle channel & resize
            slices_data = np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)

        else:
            slices_data = self._select_and_pad_slices(mri_data_3d) # Shape: (num_slices, H, W)


        # Apply transforms to each slice
        processed_slices_list = []
        for i in range(self.num_slices):
            # Get slice - Shape (H, W) numpy array
            slice_img = slices_data[i, :, :]

            # Apply base transforms (ToTensor, Resize, Normalize)
            # ToTensor expects HWC or HW. Our slice_img is HW.
            # It should convert HW -> 1HW (add channel dim) and scale [0,1]
            transformed_slice = self.base_transform(slice_img) # Output shape (C, H_resized, W_resized)

            # Apply augmentations if enabled
            if self.augment:
                transformed_slice = self.augment_transform(transformed_slice)

            processed_slices_list.append(transformed_slice)

        # Stack slices into a single tensor
        processed_slices = torch.stack(processed_slices_list) # Shape: (num_slices, C, H_resized, W_resized)


        # Ensure correct number of channels (e.g., repeat grayscale for models expecting 3 channels)
        num_channels = processed_slices.shape[1]
        # Check if model expects 3 channels (common for ImageNet pre-trained models)
        # This check might need refinement based on the specific vision encoder used later
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
             # Create a simple dummy NIfTI file (e.g., 64x64x30)
             dummy_volume = np.random.rand(64, 64, 30).astype(np.float32) * 255
             nifti_img = nib.Nifti1Image(dummy_volume, affine=np.eye(4))
             nib.save(nifti_img, file_path)
             print(f"Created dummy file: {file_path}")
        # Add the DIRECTORY path to the list, as the dataset class handles the suffix
        dummy_file_paths.append(scan_dir)


    dummy_labels = [
        {'question': 'Is there a tumor?', 'answer': 'Yes, a large enhancing tumor is present.'}, # VQA example
        'Findings: Examination reveals a well-defined lesion in the left temporal lobe. Impression: Likely glioma.' # Report gen example
    ]

    print("--- Stage 1 Example ---")
    # Create Dataset instances
    try:
        train_dataset = MRIDataset(dummy_file_paths, dummy_labels, augment=True)
        val_dataset = MRIDataset(dummy_file_paths, dummy_labels, augment=False) # No augmentation for validation

        # Test getting a single item
        print("Testing single item retrieval...")
        single_item = train_dataset[0]
        print(f"Single item keys: {single_item.keys()}")
        print(f"Single item pixel values shape: {single_item['pixel_values'].shape}") # Should be (NumSlices, Channels, ImgSize, ImgSize)
        if 'question' in single_item:
            print(f"Single item question: {single_item['question']}")
            print(f"Single item answer: {single_item['answer']}")
        if 'report' in single_item:
            print(f"Single item report: {single_item['report']}")


        # Create DataLoaders
        print("\nTesting DataLoader...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Use 0 workers for initial testing/debugging
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Get a sample batch
        sample_batch = next(iter(train_loader))
        print(f"\nSample batch loaded. Keys: {sample_batch.keys()}")
        print(f"Batch pixel values shape: {sample_batch['pixel_values'].shape}") # Should be (BatchSize, NumSlices, Channels, ImgSize, ImgSize)
        if 'question' in sample_batch:
            print(f"Batch sample question: {sample_batch['question'][0]}")
            print(f"Batch sample answer: {sample_batch['answer'][0]}")
        if 'report' in sample_batch:
            print(f"Batch sample report: {sample_batch['report'][0]}")

    except Exception as e:
        print(f"\nError during dataset/dataloader test: {e}")
        import traceback
        traceback.print_exc()


    print("\nStage 1: Data loading and preprocessing setup complete.\n")

# Clean up dummy files (optional)
# import shutil
# shutil.rmtree(dummy_dir)
# print("Cleaned up dummy data directory.")
