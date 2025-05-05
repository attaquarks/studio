# ========== Stage 1: Data Acquisition and Preprocessing ==========
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import cv2
import warnings

# --- Configuration ---
TARGET_SIZE = (224, 224) # Target dimensions for slices
IMG_SIZE = TARGET_SIZE[0] # Maintain consistent naming if needed elsewhere
NORMALIZATION = "zero_mean_unit_var" # Normalization strategy: "zero_mean_unit_var", "min_max", "none"
BATCH_SIZE = 4
NUM_WORKERS = 2
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
# MRI_TYPE_SUFFIX = 't1ce.nii.gz' # Example: Handled within file path resolution now


class MRIPreprocessor:
    """Handles the preprocessing of 3D MRI volumes into 2D slices."""

    def __init__(self, target_size=TARGET_SIZE, normalization=NORMALIZATION):
        """
        Initialize the MRI preprocessor.

        Args:
            target_size: Target dimensions for the slices
            normalization: Normalization strategy
        """
        self.target_size = target_size
        self.normalization = normalization

    def load_nifti_volume(self, file_path: str) -> np.ndarray:
        """Load a 3D MRI volume from a NIfTI file."""
        try:
            nifti_img = nib.load(file_path)
            volume_data = nifti_img.get_fdata(dtype=np.float32)
            # Ensure canonical orientation (e.g., RAS+) if possible
            # nifti_img = nib.as_closest_canonical(nifti_img)
            # volume_data = nifti_img.get_fdata(dtype=np.float32)
            return volume_data
        except FileNotFoundError:
             warnings.warn(f"NIfTI file not found: {file_path}. Returning empty array.")
             return np.array([], dtype=np.float32)
        except Exception as e:
             warnings.warn(f"Error loading NIfTI file {file_path}: {e}. Returning empty array.")
             return np.array([], dtype=np.float32)

    def extract_axial_slices(self, volume: np.ndarray, start_idx=None, end_idx=None) -> np.ndarray:
        """Extract axial slices from a 3D volume."""
        if volume.ndim != 3 or volume.size == 0:
             warnings.warn(f"Invalid volume shape {volume.shape} for slice extraction. Returning empty array.")
             return np.array([], dtype=np.float32)

        # Determine axial dimension (usually the last one, but could vary)
        # A common heuristic: the dimension with the smallest size is often the slice dim.
        # Let's assume the last dimension is axial (Z) for now. Adjust if needed.
        axial_dim_index = 2 # Assume (H, W, Z) or (X, Y, Z)
        if volume.shape[0] < volume.shape[1] and volume.shape[0] < volume.shape[2]:
             axial_dim_index = 0 # Assume (Z, H, W)
             warnings.warn("Detected Z dimension likely at index 0. Extracting slices accordingly.")
        elif volume.shape[1] < volume.shape[0] and volume.shape[1] < volume.shape[2]:
             axial_dim_index = 1 # Assume (H, Z, W) - less common
             warnings.warn("Detected Z dimension likely at index 1. Extracting slices accordingly.")

        num_slices_available = volume.shape[axial_dim_index]

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = num_slices_available

        start_idx = max(0, start_idx)
        end_idx = min(num_slices_available, end_idx)

        if start_idx >= end_idx:
             warnings.warn(f"Invalid slice range [{start_idx}, {end_idx}) for volume with {num_slices_available} slices. Returning empty array.")
             return np.array([], dtype=np.float32)

        # Extract slices along the detected axial plane
        if axial_dim_index == 2:
             # Standard (H, W, Z)
             axial_slices = volume[:, :, start_idx:end_idx] # Shape (H, W, num_selected_slices)
             # Transpose to (num_selected_slices, H, W)
             axial_slices = np.transpose(axial_slices, (2, 0, 1))
        elif axial_dim_index == 0:
             # (Z, H, W)
             axial_slices = volume[start_idx:end_idx, :, :] # Shape (num_selected_slices, H, W)
        elif axial_dim_index == 1:
             # (H, Z, W)
             axial_slices = volume[:, start_idx:end_idx, :] # Shape (H, num_selected_slices, W)
             # Transpose to (num_selected_slices, H, W)
             axial_slices = np.transpose(axial_slices, (1, 0, 2))
        else:
            # Should not happen, fallback
            warnings.warn("Could not determine axial dimension. Using default (last dim).")
            axial_slices = volume[:, :, start_idx:end_idx]
            axial_slices = np.transpose(axial_slices, (2, 0, 1))


        return np.ascontiguousarray(axial_slices) # Ensure contiguous memory

    def normalize(self, slices: np.ndarray) -> np.ndarray:
        """Normalize slice intensity values."""
        if slices.size == 0: return slices # Return empty if input is empty

        if self.normalization == "zero_mean_unit_var":
            # Zero mean, unit variance normalization
            mean = np.mean(slices)
            std = np.std(slices)
            return (slices - mean) / (std + 1e-10)
        elif self.normalization == "min_max":
            # Min-max normalization to [0, 1]
            min_val = np.min(slices)
            max_val = np.max(slices)
            return (slices - min_val) / (max_val - min_val + 1e-10)
        elif self.normalization == "none":
             return slices # No normalization
        else:
             warnings.warn(f"Unknown normalization type '{self.normalization}'. Returning original slices.")
             return slices

    def resize(self, slices: np.ndarray) -> np.ndarray:
        """Resize slices to target dimensions using cv2."""
        if slices.ndim != 3 or slices.shape[0] == 0: # Expecting (N, H, W)
             warnings.warn(f"Invalid input shape {slices.shape} for resizing. Expecting (N, H, W). Skipping resize.")
             return slices

        num_slices, h, w = slices.shape
        if (h, w) == self.target_size:
             return slices # No resize needed

        resized_slices = np.zeros((num_slices, *self.target_size), dtype=slices.dtype)
        for i, slice_img in enumerate(slices):
             # Ensure the slice is 2D before resizing
             if slice_img.ndim == 2:
                # INTER_AREA is good for downsampling, INTER_LINEAR or INTER_CUBIC for upsampling
                inter_method = cv2.INTER_AREA if (h > self.target_size[0] or w > self.target_size[1]) else cv2.INTER_LINEAR
                resized_slices[i] = cv2.resize(slice_img, self.target_size, interpolation=inter_method)
             else:
                 warnings.warn(f"Slice {i} has unexpected shape {slice_img.shape}. Skipping resize for this slice.")
                 # Optionally fill with zeros or attempt to reshape/handle
                 # For now, copy zeros to maintain shape
                 resized_slices[i] = np.zeros(self.target_size, dtype=slices.dtype)

        return resized_slices

    def process_volume(self, file_path: str) -> np.ndarray:
        """Complete processing pipeline for a single MRI volume."""
        # Load volume
        volume = self.load_nifti_volume(file_path)
        if volume.size == 0:
             # Return an empty array with expected channel dim if loading failed
             return np.array([], dtype=np.float32).reshape(0, self.target_size[0], self.target_size[1])


        # Extract axial slices
        slices = self.extract_axial_slices(volume)
        if slices.size == 0:
             return np.array([], dtype=np.float32).reshape(0, self.target_size[0], self.target_size[1])

        # Normalize
        slices = self.normalize(slices)

        # Resize
        slices = self.resize(slices)

        # Ensure float32 type
        slices = slices.astype(np.float32)

        return slices


class BraTSDataset(Dataset):
    """Dataset class for BraTS or similar MRI datasets with reports/questions."""

    def __init__(self,
                 data_dir: str,
                 annotations: pd.DataFrame, # Pass dataframe directly
                 preprocessor: MRIPreprocessor,
                 mode: str = "vqa",  # 'vqa' or 'report'
                 transform=None,
                 n_slices: int = 64): # Add n_slices parameter
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing MRI volumes
            annotations: DataFrame with annotations (questions/reports/file_name)
            preprocessor: MRI preprocessor instance
            mode: 'vqa' for question-answering or 'report' for report generation
            transform: Additional transforms for data augmentation (applied per slice)
            n_slices: Number of slices to select/pad to.
        """
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.mode = mode
        self.transform = transform
        self.n_slices = n_slices

        # Use the provided annotations dataframe
        self.annotations = annotations

        # Construct full file paths
        # Assume 'file_name' column contains relative paths or identifiers
        # that need to be joined with data_dir. Adjust if 'file_name' is absolute.
        # Handle potential missing files gracefully later in __getitem__
        self.file_paths = [os.path.join(self.data_dir, fname) for fname in self.annotations['file_name'].values]

    def __len__(self):
        return len(self.file_paths)

    def _select_and_pad_slices(self, slices: np.ndarray) -> np.ndarray:
         """Selects a subset of slices or pads to reach self.n_slices."""
         current_slices = slices.shape[0]
         target_slices = self.n_slices

         if current_slices == 0:
              # Handle case where loading/preprocessing yielded no slices
              warnings.warn("Input data has 0 slices. Padding with zeros.")
              return np.zeros((target_slices, *self.preprocessor.target_size), dtype=np.float32)

         if current_slices == target_slices:
              return slices
         elif current_slices > target_slices:
              # Select N slices (e.g., uniformly spaced or middle N)
              # Middle N slices:
              center = current_slices // 2
              start = max(0, center - target_slices // 2)
              end = start + target_slices
              # Adjust if selection goes out of bounds (shouldn't happen with max(0,..))
              selected_slices = slices[start:end, :, :]
              # Ensure the shape is correct after slicing
              if selected_slices.shape[0] != target_slices:
                    warnings.warn(f"Slice selection error: expected {target_slices}, got {selected_slices.shape[0]}. Re-padding.")
                    # Fallback to padding just in case logic failed
                    return self._select_and_pad_slices(selected_slices) # Recursive call with corrected slice attempt
              return selected_slices
         else: # current_slices < target_slices
              # Pad with zeros
              padding_size = target_slices - current_slices
              pad_before = padding_size // 2
              pad_after = padding_size - pad_before
              # np.pad format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
              padding = [(pad_before, pad_after), (0, 0), (0, 0)]
              padded_slices = np.pad(slices, padding, mode='constant', constant_values=0)
              return padded_slices

    def __getitem__(self, idx):
        # Get file path
        file_path = self.file_paths[idx]

        # Process MRI volume using the preprocessor
        # Process_volume already handles loading, slicing, normalizing, resizing
        slices = self.preprocessor.process_volume(file_path) # Shape (N_orig, H_resized, W_resized)

        # Select or pad slices to the target number (n_slices)
        slices = self._select_and_pad_slices(slices) # Shape (n_slices, H_resized, W_resized)

        # Convert to tensor and add channel dimension: (n_slices, 1, H, W)
        # We use ToTensor from torchvision.transforms which handles HWC -> CHW and scales to [0, 1]
        # However, our data is HW float. ToTensor *should* add C=1.
        # Let's prepare it as (n_slices, H, W) and let the transform handle it.
        # The transform will be applied slice-by-slice if defined.

        slices_tensor = torch.from_numpy(slices).float() # (n_slices, H, W)

        # Apply data augmentation if specified (per slice)
        if self.transform:
            # Apply transform to each slice (expecting transform to handle C dimension if needed)
            # Ensure transform inputs are suitable (e.g., PIL image or Tensor CHW)
            # If transform expects Tensor CHW, add channel dim first
            processed_slices_list = []
            for i in range(slices_tensor.shape[0]):
                 slice_hw = slices_tensor[i] # (H, W)
                 # Most torchvision transforms expect Tensor (C, H, W) or PIL Image
                 # Let's add channel dim manually here before applying transforms
                 slice_chw = slice_hw.unsqueeze(0) # (1, H, W)
                 augmented_slice = self.transform(slice_chw)
                 processed_slices_list.append(augmented_slice)
            slices_tensor = torch.stack(processed_slices_list) # (n_slices, C, H, W)
        else:
            # If no transform, just add the channel dimension
             slices_tensor = slices_tensor.unsqueeze(1) # (n_slices, 1, H, W)

        # Ensure 3 channels if needed by downstream models (e.g., repeat grayscale)
        if slices_tensor.shape[1] == 1:
             slices_tensor = slices_tensor.repeat(1, 3, 1, 1) # (n_slices, 3, H, W)


        # Get annotation data based on mode
        item = {'slices': slices_tensor} # Use 'slices' key
        annotation_row = self.annotations.iloc[idx]

        if self.mode == "vqa":
            # Check if 'question' and 'answer' columns exist
            if 'question' not in annotation_row or 'answer' not in annotation_row:
                 warnings.warn(f"Missing 'question' or 'answer' column in annotations for VQA mode at index {idx}.")
                 item['question'] = "Missing question"
                 item['answer'] = "Missing answer"
            else:
                 item['question'] = annotation_row['question']
                 item['answer'] = annotation_row['answer']
        elif self.mode == "report":
            # Check if 'report' column exists
            if 'report' not in annotation_row:
                 warnings.warn(f"Missing 'report' column in annotations for report mode at index {idx}.")
                 item['report'] = "Missing report"
            else:
                 item['report'] = annotation_row['report']
        else:
             raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vqa' or 'report'.")

        # Rename 'slices' key to 'pixel_values' if required by downstream model (e.g. HF models)
        if 'pixel_values' not in item and 'slices' in item:
            item['pixel_values'] = item.pop('slices')

        return item


class NeuroReportDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for NeuroReport."""

    def __init__(self,
                 data_dir: str,
                 annotations_path: str,
                 batch_size: int = BATCH_SIZE,
                 mode: str = "vqa",
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 val_split: float = DEFAULT_VAL_SPLIT,
                 test_split: float = DEFAULT_TEST_SPLIT,
                 num_workers: int = NUM_WORKERS,
                 normalization: str = NORMALIZATION,
                 n_slices: int = 64): # Add n_slices parameter
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing MRI volumes
            annotations_path: Path to CSV/JSON with questions/reports and 'file_name' column
            batch_size: Batch size for training
            mode: 'vqa' for question-answering or 'report' for report generation
            target_size: Target dimensions for slices
            val_split: Validation split ratio
            test_split: Test split ratio
            num_workers: Number of workers for data loading
            normalization: Normalization strategy for preprocessor
            n_slices: Number of slices to select/pad to.
        """
        super().__init__()
        self.data_dir = data_dir
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.mode = mode
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.target_size = target_size
        self.normalization = normalization
        self.n_slices = n_slices


        # Create preprocessor
        self.preprocessor = MRIPreprocessor(target_size=self.target_size, normalization=self.normalization)

        # Define data augmentation transforms (expecting Tensor CHW input)
        # Example: Convert to PIL for some transforms if needed, then back to Tensor
        # Note: Transforms like ColorJitter expect 3 channels. Ensure data is 3-channel before applying.
        self.train_transform = transforms.Compose([
            # Ensure input is Tensor (C, H, W) before these transforms
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Smaller translation
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Apply only if data is 3-channel
            # Add more transforms as needed
        ])

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_annotations = None # To store loaded annotations

    def prepare_data(self):
        """Load annotations file. Called once per node."""
        try:
            if self.annotations_path.endswith('.csv'):
                self.full_annotations = pd.read_csv(self.annotations_path)
            elif self.annotations_path.endswith('.json'):
                 # Handle different JSON orientations if needed
                 self.full_annotations = pd.read_json(self.annotations_path, orient='records')
            else:
                 raise ValueError(f"Unsupported annotation file format: {self.annotations_path}. Use .csv or .json.")
            print(f"Annotations loaded successfully from {self.annotations_path}.")
            # Basic validation
            if 'file_name' not in self.full_annotations.columns:
                 raise ValueError("'file_name' column not found in annotations.")
            if self.mode == 'vqa' and ('question' not in self.full_annotations.columns or 'answer' not in self.full_annotations.columns):
                 raise ValueError("'question' and 'answer' columns required for VQA mode.")
            if self.mode == 'report' and 'report' not in self.full_annotations.columns:
                 raise ValueError("'report' column required for report mode.")

        except FileNotFoundError:
             print(f"Error: Annotations file not found at {self.annotations_path}")
             raise
        except Exception as e:
            print(f"Error loading or validating annotations: {e}")
            raise


    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing. Called on each process (CPU/GPU)."""
        if not hasattr(self, 'full_annotations') or self.full_annotations is None:
             # Load annotations if not already loaded (e.g., in DDP scenario without prepare_data on all ranks)
             self.prepare_data()

        # Ensure annotations are available
        if self.full_annotations is None:
            raise RuntimeError("Annotations were not loaded correctly.")

        # Split data indices
        num_samples = len(self.full_annotations)
        indices = list(range(num_samples))

        if self.val_split + self.test_split >= 1.0:
             raise ValueError("val_split + test_split must be less than 1.0")

        # Split train and temp (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(self.val_split + self.test_split),
            random_state=42 # Use a fixed seed for reproducibility
        )

        # Split temp into val and test
        # Adjust test_size calculation for the second split
        relative_test_size = self.test_split / (self.val_split + self.test_split) if (self.val_split + self.test_split) > 0 else 0

        if relative_test_size > 0 and len(temp_idx) > 0:
             val_idx, test_idx = train_test_split(
                 temp_idx,
                 test_size=relative_test_size,
                 random_state=42 # Use the same seed for consistency
             )
        elif len(temp_idx) > 0: # Only validation split needed
            val_idx = temp_idx
            test_idx = []
        else: # No validation or test split
            val_idx = []
            test_idx = []

        # Create annotation subsets using iloc for DataFrames
        train_annotations = self.full_annotations.iloc[train_idx].reset_index(drop=True)
        val_annotations = self.full_annotations.iloc[val_idx].reset_index(drop=True)
        test_annotations = self.full_annotations.iloc[test_idx].reset_index(drop=True)


        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            self.train_dataset = BraTSDataset(
                self.data_dir,
                train_annotations, # Pass dataframe subset
                self.preprocessor,
                mode=self.mode,
                transform=self.train_transform,
                n_slices=self.n_slices
            )
            print(f"Train dataset created with {len(self.train_dataset)} samples.")

            self.val_dataset = BraTSDataset(
                self.data_dir,
                val_annotations, # Pass dataframe subset
                self.preprocessor,
                mode=self.mode,
                transform=None, # No augmentation for validation
                n_slices=self.n_slices
            )
            print(f"Validation dataset created with {len(self.val_dataset)} samples.")

        if stage == 'test' or stage is None:
            self.test_dataset = BraTSDataset(
                self.data_dir,
                test_annotations, # Pass dataframe subset
                self.preprocessor,
                mode=self.mode,
                transform=None, # No augmentation for test
                n_slices=self.n_slices
            )
            print(f"Test dataset created with {len(self.test_dataset)} samples.")

        if stage == 'predict' or stage is None:
            # Often predict uses the test set, or a separate prediction set
            if not self.test_dataset: # Create if not already done
                self.test_dataset = BraTSDataset(
                    self.data_dir, test_annotations, self.preprocessor, mode=self.mode, n_slices=self.n_slices
                )
            self.predict_dataset = self.test_dataset # Use test set for prediction example
            print(f"Predict dataset created with {len(self.predict_dataset)} samples (using test set).")


    def train_dataloader(self):
        if not self.train_dataset: self.setup('fit') # Ensure setup is called
        if not self.train_dataset: raise RuntimeError("Train dataset not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True # Drop last incomplete batch for consistent batch sizes
        )

    def val_dataloader(self):
        if not self.val_dataset: self.setup('fit') # Ensure setup is called
        if not self.val_dataset: raise RuntimeError("Validation dataset not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if not self.test_dataset: self.setup('test') # Ensure setup is called
        if not self.test_dataset: raise RuntimeError("Test dataset not initialized.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        if not self.predict_dataset: self.setup('predict') # Ensure setup is called
        if not self.predict_dataset: raise RuntimeError("Predict dataset not initialized.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# --- Example Usage (within main script or separate file) ---
if __name__ == "__main__":
    print("--- Stage 1 Example ---")
    # Create dummy data and annotations for demonstration
    DUMMY_DATA_DIR = "./dummy_mri_data_s1"
    DUMMY_ANNOTATIONS_FILE = "./dummy_annotations_s1.csv"
    NUM_DUMMY_SAMPLES = 20
    IMG_H, IMG_W, IMG_D = 64, 64, 30 # Small dummy volume size

    os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
    dummy_data = []
    for i in range(NUM_DUMMY_SAMPLES):
        scan_name = f"scan_{i+1}.nii.gz"
        file_path = os.path.join(DUMMY_DATA_DIR, scan_name)
        # Create dummy NIfTI file
        dummy_volume = np.random.rand(IMG_H, IMG_W, IMG_D).astype(np.float32) * 255
        nifti_img = nib.Nifti1Image(dummy_volume, affine=np.eye(4))
        nib.save(nifti_img, file_path)
        # Add annotation entry
        dummy_data.append({
            'file_name': scan_name,
            'question': f'What is seen in scan {i+1}?',
            'answer': f'Findings consistent with normal scan {i+1}.',
            'report': f'Report for scan {i+1}: No acute abnormalities detected.'
        })
    dummy_annotations_df = pd.DataFrame(dummy_data)
    dummy_annotations_df.to_csv(DUMMY_ANNOTATIONS_FILE, index=False)
    print(f"Created dummy data in {DUMMY_DATA_DIR} and annotations in {DUMMY_ANNOTATIONS_FILE}")


    # Instantiate DataModule
    data_module = NeuroReportDataModule(
        data_dir=DUMMY_DATA_DIR,
        annotations_path=DUMMY_ANNOTATIONS_FILE,
        batch_size=BATCH_SIZE,
        mode="vqa", # or "report"
        target_size=TARGET_SIZE,
        num_workers=0 # Set to 0 for easier debugging in main process
    )

    # Prepare and setup data
    try:
        data_module.prepare_data()
        data_module.setup('fit') # Setup for training/validation

        # Get a sample batch from train loader
        train_loader = data_module.train_dataloader()
        sample_batch = next(iter(train_loader))

        print("\nSample batch loaded.")
        print(f"  Keys: {sample_batch.keys()}")
        print(f"  Pixel values shape: {sample_batch['pixel_values'].shape}") # B, N_slices, C, H, W
        print(f"  Pixel values dtype: {sample_batch['pixel_values'].dtype}")
        if 'question' in sample_batch:
            print(f"  Sample question: {sample_batch['question'][0]}")
            print(f"  Sample answer: {sample_batch['answer'][0]}")
        if 'report' in sample_batch:
            print(f"  Sample report: {sample_batch['report'][0]}")

        # Test val loader
        val_loader = data_module.val_dataloader()
        val_batch = next(iter(val_loader))
        print("\nValidation batch sample pixel values shape:", val_batch['pixel_values'].shape)

    except Exception as e:
        print(f"\nError during DataModule example usage: {e}")
        import traceback
        traceback.print_exc()

    # Clean up dummy files (optional)
    # import shutil
    # try:
    #     if os.path.exists(DUMMY_DATA_DIR): shutil.rmtree(DUMMY_DATA_DIR)
    #     if os.path.exists(DUMMY_ANNOTATIONS_FILE): os.remove(DUMMY_ANNOTATIONS_FILE)
    #     if os.path.exists('train_annotations.csv'): os.remove('train_annotations.csv')
    #     if os.path.exists('val_annotations.csv'): os.remove('val_annotations.csv')
    #     if os.path.exists('test_annotations.csv'): os.remove('test_annotations.csv')
    #     print("Cleaned up dummy files.")
    # except Exception as e_clean: print(f"Error cleaning up: {e_clean}")

    print("\nStage 1: Data loading and preprocessing setup complete using DataModule.\n")


# Expose BATCH_SIZE for other stages if needed (can be accessed via data_module instance)
NUM_SLICES_PER_SCAN = 64 # Set explicitly or derive from DataModule if needed later

# Constants derived/used in this stage
# TARGET_SIZE, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_SLICES_PER_SCAN
