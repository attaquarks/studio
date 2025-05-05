# ========== Stage 1: Data Acquisition and Preprocessing ==========
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split # Re-added for splitting
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import cv2
import warnings
import glob # For finding files

# --- Configuration ---
TARGET_SIZE = (224, 224) # Target dimensions for slices
IMG_SIZE = TARGET_SIZE[0] # Maintain consistent naming if needed elsewhere
NORMALIZATION = "zero_mean_unit_var" # Normalization strategy: "zero_mean_unit_var", "min_max", "none"
BATCH_SIZE = 4
NUM_WORKERS = 2
VAL_SPLIT = 0.1 # Re-enabled validation split
TEST_SPLIT = 0.1 # Re-enabled test split
# MRI_TYPE_SUFFIX_TO_LOAD = 't1ce.nii.gz' # Specific modality to load from BraTS folders
MRI_MODALITIES_TO_LOAD = ['t1ce', 'flair'] # Load multiple modalities example
NUM_SLICES_PER_SCAN = 64 # Example: Can be derived from DataModule if needed later

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
    """
    Dataset class for BraTS MRI dataset, handling multiple modalities and optional annotations.
    """
    def __init__(self,
                 data_dir: str,
                 annotations: Optional[pd.DataFrame], # Make annotations optional or handle cases without it
                 preprocessor: MRIPreprocessor,
                 modalities: List[str] = MRI_MODALITIES_TO_LOAD, # Modalities to load (e.g., ['t1ce', 'flair'])
                 mode: str = "report",  # Default to 'report' if annotations might be missing
                 transform=None,
                 n_slices: int = NUM_SLICES_PER_SCAN): # Use constant from config
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing BraTS patient folders (e.g., 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX')
            annotations: DataFrame with annotations mapping 'patient_id' to 'report'/'question'/'answer'. Optional.
            preprocessor: MRI preprocessor instance
            modalities: List of MRI modalities to load (e.g., 't1ce', 'flair', 't1', 't2').
            mode: 'vqa' or 'report'. Determines which annotation columns are expected if annotations are provided.
            transform: Additional transforms for data augmentation (applied per slice per modality).
            n_slices: Number of slices to select/pad to.
        """
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.modalities = modalities
        self.mode = mode
        self.transform = transform
        self.n_slices = n_slices
        self.annotations = annotations

        # Find all patient directories directly within the data_dir
        # Assumes structure like data_dir/BraTS20_Training_XXX/BraTS20_Training_XXX_modality.nii.gz
        self.patient_dirs = [d for d in glob.glob(os.path.join(data_dir, "BraTS20_*")) if os.path.isdir(d)]

        if not self.patient_dirs:
             # Check one level deeper if the initial glob fails (e.g., if data_dir points to the top-level folder)
             potential_subdir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData') # Common structure
             if os.path.isdir(potential_subdir):
                  self.patient_dirs = [d for d in glob.glob(os.path.join(potential_subdir, "BraTS20_*")) if os.path.isdir(d)]

        if not self.patient_dirs:
             raise FileNotFoundError(f"No patient directories (BraTS20_*) found in {data_dir} or its common subdirectories.")

        print(f"Found {len(self.patient_dirs)} patient directories in {data_dir}.")

        # Create a mapping from patient ID to directory path
        self.patient_id_to_path = {os.path.basename(p): p for p in self.patient_dirs}

        # Filter patient paths based on available annotations if provided
        if self.annotations is not None and 'patient_id' in self.annotations.columns:
            valid_patient_ids = set(self.annotations['patient_id'].unique())
            self.patient_dirs = [p for p in self.patient_dirs if os.path.basename(p) in valid_patient_ids]
            print(f"Filtered down to {len(self.patient_dirs)} patients based on annotations.")
            if not self.patient_dirs:
                 raise ValueError("No patient directories match the patient_ids found in the annotations file.")
        elif self.annotations is not None and 'patient_id' not in self.annotations.columns:
            warnings.warn("'patient_id' column not found in annotations. Cannot filter patients based on annotations.")
            # Proceed using all found patient directories


    def __len__(self):
        return len(self.patient_dirs)

    def _find_modality_file(self, patient_dir: str, patient_id: str, modality: str) -> Optional[str]:
        """Finds the NIfTI file for a specific modality within a patient directory."""
        # Example file pattern: BraTS20_Training_XXX_modality.nii.gz
        file_pattern = os.path.join(patient_dir, f"{patient_id}_{modality}.nii.gz")
        files = glob.glob(file_pattern)
        if files:
            return files[0]
        else:
            warnings.warn(f"Modality file not found for pattern: {file_pattern}")
            return None

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
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        # Load and process slices for each requested modality
        processed_modalities = []
        for modality in self.modalities:
            modality_file = self._find_modality_file(patient_dir, patient_id, modality)
            if modality_file:
                 slices = self.preprocessor.process_volume(modality_file) # (N_orig, H, W)
                 slices = self._select_and_pad_slices(slices) # (n_slices, H, W)
                 processed_modalities.append(slices)
            else:
                 # Handle missing modality (e.g., pad with zeros)
                 warnings.warn(f"Missing modality '{modality}' for patient {patient_id}. Padding with zeros.")
                 processed_modalities.append(np.zeros((self.n_slices, *self.preprocessor.target_size), dtype=np.float32))

        # Stack modalities along the channel dimension
        # Input shape: (num_modalities, n_slices, H, W)
        slices_np = np.stack(processed_modalities, axis=0)

        # Convert to tensor: (C, S, H, W) -> needs rearrange for ViT: (S, C, H, W)
        slices_tensor = torch.from_numpy(slices_np).float()
        # Rearrange: (C, S, H, W) -> (S, C, H, W)
        slices_tensor = slices_tensor.permute(1, 0, 2, 3)

        # Apply data augmentation if specified (applied identically across channels for each slice)
        if self.transform:
            augmented_slices_list = []
            for i in range(slices_tensor.shape[0]): # Iterate through slices
                 slice_chw = slices_tensor[i] # (C, H, W)
                 augmented_slice = self.transform(slice_chw)
                 augmented_slices_list.append(augmented_slice)
            slices_tensor = torch.stack(augmented_slices_list) # (S, C, H, W)

        # Ensure 3 channels if needed (e.g., for standard ViT)
        # If only 1 modality loaded -> repeat. If 2 -> pad? If > 3 -> slice?
        # Current ViT model expects 3 channels. Let's handle common cases.
        num_channels = slices_tensor.shape[1]
        if num_channels == 1:
            slices_tensor = slices_tensor.repeat(1, 3, 1, 1) # (S, 3, H, W)
        elif num_channels == 2:
             # Example: Pad with a zero channel
             zeros = torch.zeros_like(slices_tensor[:, :1, :, :]) # (S, 1, H, W)
             slices_tensor = torch.cat([slices_tensor, zeros], dim=1) # (S, 3, H, W)
        elif num_channels > 3:
             warnings.warn(f"Input has {num_channels} channels. Taking the first 3.")
             slices_tensor = slices_tensor[:, :3, :, :] # (S, 3, H, W)
        # If num_channels == 3, do nothing.

        # --- Prepare Output ---
        item = {'pixel_values': slices_tensor} # Key often used by vision models

        # Add annotations if available
        if self.annotations is not None:
            annotation_row = self.annotations[self.annotations['patient_id'] == patient_id]
            if not annotation_row.empty:
                 annotation_row = annotation_row.iloc[0] # Get first match
                 if self.mode == "vqa":
                      if 'question' not in annotation_row or 'answer' not in annotation_row:
                           warnings.warn(f"Missing 'question' or 'answer' column in annotations for VQA mode at index {idx}.")
                           item['question'] = "Missing question"
                           item['answer'] = "Missing answer"
                      else:
                           item['question'] = annotation_row['question']
                           item['answer'] = annotation_row['answer']
                 elif self.mode == "report":
                      if 'report' not in annotation_row:
                           warnings.warn(f"Missing 'report' column in annotations for report mode at index {idx}.")
                           item['report'] = "Missing report"
                      else:
                           item['report'] = annotation_row['report']
            else:
                 warnings.warn(f"No annotation found for patient_id {patient_id} at index {idx}. Returning image data only.")
                 if self.mode == "vqa": item['question'], item['answer'] = "Missing", "Missing"
                 if self.mode == "report": item['report'] = "Missing"

        # If no annotations, item only contains 'pixel_values'
        elif self.mode == "vqa": item['question'], item['answer'] = "Missing", "Missing"
        elif self.mode == "report": item['report'] = "Missing"


        return item


class NeuroReportDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for NeuroReport with BraTS."""

    def __init__(self,
                 data_dir: str,
                 annotations_path: Optional[str], # Annotations are optional now
                 batch_size: int = BATCH_SIZE,
                 mode: str = "report", # Default to report if no annotations assumed
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 val_split: float = VAL_SPLIT, # Re-enabled
                 test_split: float = TEST_SPLIT, # Re-enabled
                 num_workers: int = NUM_WORKERS,
                 normalization: str = NORMALIZATION,
                 n_slices: int = NUM_SLICES_PER_SCAN,
                 modalities: List[str] = MRI_MODALITIES_TO_LOAD):
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing BraTS patient folders.
            annotations_path: Path to CSV/JSON with annotations (optional). Must contain 'patient_id' if provided.
            batch_size: Batch size for training.
            mode: 'vqa' or 'report'.
            target_size: Target dimensions for slices.
            val_split: Validation split ratio (re-enabled).
            test_split: Test split ratio (re-enabled).
            num_workers: Number of workers for data loading.
            normalization: Normalization strategy for preprocessor.
            n_slices: Number of slices to select/pad to.
            modalities: List of modalities to load.
        """
        super().__init__()
        self.data_dir = data_dir
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.mode = mode
        self.val_split = val_split # Re-enabled
        self.test_split = test_split # Re-enabled
        self.num_workers = num_workers
        self.target_size = target_size
        self.normalization = normalization
        self.n_slices = n_slices
        self.modalities = modalities
        self._seed = 42 # For reproducibility of splits

        # Create preprocessor
        self.preprocessor = MRIPreprocessor(target_size=self.target_size, normalization=self.normalization)

        # Define data augmentation transforms (expecting Tensor CHW input)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Apply only if data is 3-channel
        ])

        # Placeholders for datasets and indices
        self.train_dataset = None
        self.val_dataset = None # Re-enabled
        self.test_dataset = None # Re-enabled
        self.full_annotations = None # To store loaded annotations

    def prepare_data(self):
        """Load annotations file if provided. Called once per node."""
        if self.annotations_path:
             try:
                 if self.annotations_path.endswith('.csv'):
                     self.full_annotations = pd.read_csv(self.annotations_path)
                 elif self.annotations_path.endswith('.json'):
                      self.full_annotations = pd.read_json(self.annotations_path, orient='records')
                 else:
                      raise ValueError(f"Unsupported annotation file format: {self.annotations_path}. Use .csv or .json.")
                 print(f"Annotations loaded successfully from {self.annotations_path}.")
                 # Basic validation
                 if 'patient_id' not in self.full_annotations.columns:
                      raise ValueError("'patient_id' column not found in annotations.")
                 if self.mode == 'vqa' and ('question' not in self.full_annotations.columns or 'answer' not in self.full_annotations.columns):
                      raise ValueError("'question' and 'answer' columns required for VQA mode.")
                 if self.mode == 'report' and 'report' not in self.full_annotations.columns:
                      raise ValueError("'report' column required for report mode.")

             except FileNotFoundError: print(f"Warning: Annotations file not found at {self.annotations_path}. Proceeding without annotations.") ; self.full_annotations = None
             except Exception as e: print(f"Error loading or validating annotations: {e}") ; self.full_annotations = None
        else:
             print("No annotations path provided. Datasets will only contain images.")
             self.full_annotations = None


    def setup(self, stage: Optional[str] = None):
        """Set up datasets with train/val/test splits."""
        # Create the full dataset instance
        full_dataset = BraTSDataset(
            self.data_dir,
            self.full_annotations, # Pass loaded annotations (can be None)
            self.preprocessor,
            modalities=self.modalities,
            mode=self.mode,
            # Apply train transform only to the training split later
            transform=None, # Don't apply augmentation to the full dataset yet
            n_slices=self.n_slices
        )

        num_samples = len(full_dataset)
        indices = list(range(num_samples))

        # Calculate split sizes
        num_test = int(np.floor(self.test_split * num_samples))
        num_val = int(np.floor(self.val_split * (num_samples - num_test))) # Split val from remaining
        num_train = num_samples - num_val - num_test

        if num_train <= 0 or num_val < 0 or num_test < 0:
             raise ValueError(f"Invalid dataset split resulting in non-positive set sizes: Train={num_train}, Val={num_val}, Test={num_test}")

        # Split indices
        train_indices, temp_indices = train_test_split(indices, train_size=num_train, random_state=self._seed)
        # Split remaining into val and test
        if num_val > 0 and num_test > 0:
             val_indices, test_indices = train_test_split(temp_indices, train_size=num_val, random_state=self._seed)
        elif num_val > 0:
            val_indices = temp_indices
            test_indices = []
        elif num_test > 0:
            test_indices = temp_indices
            val_indices = []
        else: # Only training data
            val_indices, test_indices = [], []


        print(f"Dataset split: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")

        # Create Subset datasets
        if stage == 'fit' or stage is None:
             # Create train dataset WITH transform
             self.train_dataset = Subset(
                 BraTSDataset( # Recreate dataset with transform for training subset
                     self.data_dir, self.full_annotations, self.preprocessor,
                     modalities=self.modalities, mode=self.mode,
                     transform=self.train_transform, n_slices=self.n_slices
                 ),
                 train_indices
             )
             # Create val dataset WITHOUT transform
             self.val_dataset = Subset(
                 BraTSDataset(
                      self.data_dir, self.full_annotations, self.preprocessor,
                      modalities=self.modalities, mode=self.mode,
                      transform=None, n_slices=self.n_slices
                 ),
                 val_indices
             ) if val_indices else None

        if stage == 'test' or stage is None:
             # Create test dataset WITHOUT transform
             self.test_dataset = Subset(
                 BraTSDataset(
                      self.data_dir, self.full_annotations, self.preprocessor,
                      modalities=self.modalities, mode=self.mode,
                      transform=None, n_slices=self.n_slices
                 ),
                 test_indices
            ) if test_indices else None

        if stage == 'predict' or stage is None:
            # Create predict dataset (often same as test, or a separate set)
             self.predict_dataset = Subset(
                 BraTSDataset(
                      self.data_dir, self.full_annotations, self.preprocessor,
                      modalities=self.modalities, mode=self.mode,
                      transform=None, n_slices=self.n_slices
                 ),
                 test_indices # Example: use test set for prediction
            ) if test_indices else None


    def train_dataloader(self):
        if not self.train_dataset: self.setup('fit') # Ensure setup is called
        if not self.train_dataset: raise RuntimeError("Train dataset not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    # Re-enabled val_dataloader and test_dataloader
    def val_dataloader(self):
        if not self.val_dataset: self.setup('fit') # Ensure setup is called
        if not self.val_dataset:
             warnings.warn("Validation dataset not available (val_split might be 0). Returning None.")
             return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        if not self.test_dataset: self.setup('test') # Ensure setup is called
        if not self.test_dataset:
             warnings.warn("Test dataset not available (test_split might be 0). Returning None.")
             return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def predict_dataloader(self):
        if not self.predict_dataset: self.setup('predict') # Ensure setup is called
        if not self.predict_dataset:
             warnings.warn("Predict dataset not available (test_split might be 0). Returning None.")
             return None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


# --- Example Usage (within main script or separate file) ---
if __name__ == "__main__":
    print("--- Stage 1 Example (BraTS focus with splits) ---")
    # IMPORTANT: Update these paths to your downloaded dataset and annotations
    # DUMMY_DATA_DIR should point to the directory containing BraTS20_Training_XXX folders
    DUMMY_DATA_DIR = "./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" # Adjust this path
    # DUMMY_ANNOTATIONS_FILE should be the path to your CSV/JSON annotations
    # Create a dummy one if you don't have one yet, mapping patient IDs to reports/QA pairs
    DUMMY_ANNOTATIONS_FILE = "./dummy_brats_annotations.csv" # Adjust or set to None

    # Create dummy annotations if file doesn't exist
    if not os.path.exists(DUMMY_ANNOTATIONS_FILE):
        print(f"Creating dummy annotations file: {DUMMY_ANNOTATIONS_FILE}")
        # List some patient IDs found in the dummy data dir for the example
        example_patient_ids = [os.path.basename(p) for p in glob.glob(os.path.join(DUMMY_DATA_DIR, "BraTS20_Training_*"))[:20]] # Get more IDs for splitting
        if not example_patient_ids:
             print(f"Warning: Could not find example patient IDs in {DUMMY_DATA_DIR} for dummy annotations.")
             # Create generic IDs if none found
             example_patient_ids = [f"BraTS20_Training_{i:03}" for i in range(1,21)]

        dummy_annot_data = []
        for i, p_id in enumerate(example_patient_ids):
             dummy_annot_data.append({
                 'patient_id': p_id, # MUST match folder names like BraTS20_Training_XXX
                 'question': f'What is seen in scan {p_id}?',
                 'answer': f'Findings consistent with normal scan {p_id}.',
                 'report': f'Report for scan {p_id}: No acute abnormalities detected.'
             })
        dummy_annotations_df = pd.DataFrame(dummy_annot_data)
        dummy_annotations_df.to_csv(DUMMY_ANNOTATIONS_FILE, index=False)
        print(f"Created dummy annotations with patient IDs: {[d['patient_id'] for d in dummy_annot_data]}")

    # Instantiate DataModule
    data_module = NeuroReportDataModule(
        data_dir=DUMMY_DATA_DIR,
        annotations_path=DUMMY_ANNOTATIONS_FILE, # Use None if no annotations
        batch_size=2, # Smaller batch for example
        mode="report", # or "vqa"
        target_size=TARGET_SIZE,
        val_split=VAL_SPLIT, # Use configured splits
        test_split=TEST_SPLIT,
        num_workers=0, # Set to 0 for easier debugging in main process
        n_slices=NUM_SLICES_PER_SCAN,
        modalities=['t1ce', 'flair'] # Example modalities
    )

    # Prepare and setup data
    try:
        data_module.prepare_data()
        data_module.setup('fit') # Setup for training (creates train/val splits)
        data_module.setup('test') # Setup for testing (creates test split)

        # Get sample batches from loaders
        print("\n--- DataLoader Examples ---")
        train_loader = data_module.train_dataloader()
        if train_loader:
             print(f"Train loader has {len(train_loader)} batches.")
             sample_batch_train = next(iter(train_loader))
             print("Sample train batch loaded.")
             print(f"  Keys: {sample_batch_train.keys()}")
             print(f"  Pixel values shape: {sample_batch_train['pixel_values'].shape}")
             if 'report' in sample_batch_train: print(f"  Sample report: {sample_batch_train['report'][0][:50]}...")
        else: print("Train loader is None.")

        val_loader = data_module.val_dataloader()
        if val_loader:
             print(f"\nVal loader has {len(val_loader)} batches.")
             sample_batch_val = next(iter(val_loader))
             print("Sample val batch loaded.")
             print(f"  Keys: {sample_batch_val.keys()}")
             print(f"  Pixel values shape: {sample_batch_val['pixel_values'].shape}")
        else: print("Val loader is None.")

        test_loader = data_module.test_dataloader()
        if test_loader:
             print(f"\nTest loader has {len(test_loader)} batches.")
             sample_batch_test = next(iter(test_loader))
             print("Sample test batch loaded.")
             print(f"  Keys: {sample_batch_test.keys()}")
             print(f"  Pixel values shape: {sample_batch_test['pixel_values'].shape}")
        else: print("Test loader is None.")


    except FileNotFoundError as e:
         print(f"\nError: Data directory or a required file not found: {e}")
         print(f"Please ensure '{DUMMY_DATA_DIR}' points to the correct BraTS dataset location.")
    except Exception as e:
        print(f"\nError during DataModule example usage: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 1: BraTS data loading and preprocessing setup complete.\n")
