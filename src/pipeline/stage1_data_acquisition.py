# ========== Stage 1: Data Acquisition and Preprocessing ==========
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
# Removing train_test_split as per user request
# from sklearn.model_selection import train_test_split
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
# VAL_SPLIT = 0.0 # Disabled as per user request
# TEST_SPLIT = 0.0 # Disabled as per user request
# MRI_TYPE_SUFFIX_TO_LOAD = 't1ce.nii.gz' # Specific modality to load from BraTS folders
MRI_MODALITIES_TO_LOAD = ['t1ce', 'flair'] # Load multiple modalities example


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
                 n_slices: int = 64):
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
            annotation_row = self.annotations[self.annotations['patient_id'] == patient_id].iloc[0] # Get first match
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
        # If no annotations, item only contains 'pixel_values'

        return item


class NeuroReportDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for NeuroReport with BraTS."""

    def __init__(self,
                 data_dir: str,
                 annotations_path: Optional[str], # Annotations are optional now
                 batch_size: int = BATCH_SIZE,
                 mode: str = "report", # Default to report if no annotations assumed
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 # val_split: float = VAL_SPLIT, # Disabled
                 # test_split: float = TEST_SPLIT, # Disabled
                 num_workers: int = NUM_WORKERS,
                 normalization: str = NORMALIZATION,
                 n_slices: int = 64,
                 modalities: List[str] = MRI_MODALITIES_TO_LOAD):
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing BraTS patient folders.
            annotations_path: Path to CSV/JSON with annotations (optional). Must contain 'patient_id' if provided.
            batch_size: Batch size for training.
            mode: 'vqa' or 'report'.
            target_size: Target dimensions for slices.
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
        # self.val_split = val_split # Disabled
        # self.test_split = test_split # Disabled
        self.num_workers = num_workers
        self.target_size = target_size
        self.normalization = normalization
        self.n_slices = n_slices
        self.modalities = modalities

        # Create preprocessor
        self.preprocessor = MRIPreprocessor(target_size=self.target_size, normalization=self.normalization)

        # Define data augmentation transforms (expecting Tensor CHW input)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Apply only if data is 3-channel
        ])

        # Placeholders for datasets
        self.train_dataset = None
        # self.val_dataset = None # Disabled
        # self.test_dataset = None # Disabled
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
        """Set up datasets. Uses all data for training as requested."""
        if stage == 'fit' or stage is None:
            self.train_dataset = BraTSDataset(
                self.data_dir,
                self.full_annotations, # Pass loaded annotations (can be None)
                self.preprocessor,
                modalities=self.modalities,
                mode=self.mode,
                transform=self.train_transform,
                n_slices=self.n_slices
            )
            print(f"Train dataset created with {len(self.train_dataset)} samples (using all available data).")

        # No validation or test split based on user request
        # if stage == 'validate' or stage is None: # Or 'fit' stage if val during training
        #     # Create val_dataset using a split if needed, or None
        #     self.val_dataset = None # Explicitly set to None

        # if stage == 'test' or stage is None:
        #     # Create test_dataset using a split if needed, or None
        #     self.test_dataset = None # Explicitly set to None

        # if stage == 'predict' or stage is None:
        #     # Typically uses test or a separate predict set
        #     self.predict_dataset = None # Set appropriately if prediction needed


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

    # Remove val_dataloader and test_dataloader as per request
    # def val_dataloader(self):
    #     # Return None or raise error if validation is attempted
    #     return None

    # def test_dataloader(self):
    #     # Return None or raise error if testing is attempted
    #     return None

    # def predict_dataloader(self):
    #     # Return None or set up if needed
    #     return None

# --- Example Usage (within main script or separate file) ---
if __name__ == "__main__":
    print("--- Stage 1 Example (BraTS focus) ---")
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
        example_patient_ids = [os.path.basename(p) for p in glob.glob(os.path.join(DUMMY_DATA_DIR, "BraTS20_Training_*"))[:5]]
        if not example_patient_ids:
             print(f"Warning: Could not find example patient IDs in {DUMMY_DATA_DIR} for dummy annotations.")
             # Create generic IDs if none found
             example_patient_ids = [f"BraTS20_Training_{i:03}" for i in range(1,6)]

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
        num_workers=0, # Set to 0 for easier debugging in main process
        n_slices=NUM_SLICES_PER_SCAN,
        modalities=['t1ce', 'flair'] # Example modalities
    )

    # Prepare and setup data
    try:
        data_module.prepare_data()
        data_module.setup('fit') # Setup for training

        # Get a sample batch from train loader
        train_loader = data_module.train_dataloader()
        print(f"Train loader has {len(train_loader)} batches.")
        sample_batch = next(iter(train_loader))

        print("\nSample batch loaded.")
        print(f"  Keys: {sample_batch.keys()}")
        # Shape is now (Batch, Slices, Channels=3, H, W) due to modality stacking/padding
        print(f"  Pixel values shape: {sample_batch['pixel_values'].shape}")
        print(f"  Pixel values dtype: {sample_batch['pixel_values'].dtype}")
        if 'question' in sample_batch:
            print(f"  Sample question: {sample_batch['question'][0]}")
            print(f"  Sample answer: {sample_batch['answer'][0]}")
        if 'report' in sample_batch:
            print(f"  Sample report: {sample_batch['report'][0]}")


    except FileNotFoundError as e:
         print(f"\nError: Data directory or a required file not found: {e}")
         print(f"Please ensure '{DUMMY_DATA_DIR}' points to the correct BraTS dataset location.")
    except Exception as e:
        print(f"\nError during DataModule example usage: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 1: BraTS data loading and preprocessing setup complete.\n")


# Expose config derived/used in this stage
NUM_SLICES_PER_SCAN = 64 # Example: Can be derived from DataModule if needed later

```
  </change>
  <change>
    <file>src/pipeline/stage6_training.py</file>
    <description>Adapt NeuroReportModel __init__ and _prepare_batch to handle potentially missing annotations and remove validation steps as per the data split removal.</description>
    <content><![CDATA[# ========== Stage 6: Training and Fine-Tuning ==========
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader # For dummy loader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler # For dummy/scheduler
import warnings
import os
from typing import Optional

# --- Import Components from Previous Stages ---
pipeline_dir = os.path.dirname(__file__)

# Use try-except blocks for robustness if run standalone
try:
    # Import DataModule definition, but not specific dataset class directly needed here
    from .stage1_data_acquisition import NeuroReportDataModule, BATCH_SIZE, TARGET_SIZE, NUM_SLICES_PER_SCAN
except ImportError:
    warnings.warn("Could not import from stage1. Defining dummy components for Stage 6 structure.")
    BATCH_SIZE = 2
    NUM_SLICES_PER_SCAN = 8
    TARGET_SIZE=(32, 32) # Small dummy size
    class NeuroReportDataModule(pl.LightningDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.batch_size = BATCH_SIZE
        def setup(self, stage=None): pass # Minimal setup
        def train_dataloader(self): # Need a dummy loader for structure
            class DummyDataset(torch.utils.data.Dataset):
                def __init__(self): self.len=10
                def __len__(self): return self.len
                def __getitem__(self, idx):
                     # Return only pixel_values if annotations might be missing
                     return {'pixel_values': torch.randn(NUM_SLICES_PER_SCAN, 3, TARGET_SIZE[0], TARGET_SIZE[1])}
            return DataLoader(DummyDataset(), batch_size=self.batch_size)
        # def val_dataloader(self): return None # No validation loader needed now

try:
    from .stage2_vision_encoder import VisionEncoder, VISION_FEATURE_DIM as STAGE2_VISION_FEATURE_DIM
except ImportError:
    warnings.warn("Could not import VisionEncoder from stage2. Defining dummy.")
    STAGE2_VISION_FEATURE_DIM = 768
    class VisionEncoder(nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.feature_dim = STAGE2_VISION_FEATURE_DIM; self.dummy = nn.Linear(10, self.feature_dim)
        def forward(self, x): b, s, c, h, w = x.shape; return torch.randn(b, s, self.feature_dim, device=x.device)

try:
    from .stage3_slice_aggregation import SliceAggregator, AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
except ImportError:
    warnings.warn("Could not import SliceAggregator from stage3. Defining dummy.")
    STAGE3_AGGREGATOR_OUTPUT_DIM = STAGE2_VISION_FEATURE_DIM # Assume mean pooling
    class SliceAggregator(nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.output_dim = STAGE3_AGGREGATOR_OUTPUT_DIM
        def forward(self, x, **kwargs): return x.mean(dim=1)

try:
    # Bridge might be optional or Identity, handle its potential absence or type
    from .stage4_vision_language_bridge import VisionLanguageBridge # Import the bridge class
    # AGGREGATED_FEATURE_DIM needed for bridge init comes from Stage 3
    # TARGET_LANGUAGE_MODEL_DIM needed for bridge init comes from Stage 5
except ImportError:
     warnings.warn("Could not import VisionLanguageBridge from stage4. Assuming Identity bridge.")
     VisionLanguageBridge = nn.Identity # Use Identity if bridge file/class is missing

try:
    from .stage5_language_decoder import LanguageDecoder, LANGUAGE_MODEL_NAME, model_type as STAGE5_LM_TYPE, USE_LORA as STAGE5_USE_LORA, LANGUAGE_MODEL_DIM as STAGE5_LM_DIM
    TARGET_LANGUAGE_MODEL_DIM = STAGE5_LM_DIM # Use the dimension from the loaded LM
except ImportError as e:
    warnings.warn(f"Could not import components from stage5 ({e}). Defining dummy components for Stage 6 structure.")
    LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Use a consistent model name
    STAGE5_LM_TYPE = 'seq2seq'
    STAGE5_USE_LORA = False
    TARGET_LANGUAGE_MODEL_DIM = 768 # Reset based on dummy T5
    STAGE5_LM_DIM = TARGET_LANGUAGE_MODEL_DIM
    # Define dummy LanguageDecoder class
    class LanguageDecoder(nn.Module):
         def __init__(self, *args, **kwargs):
             super().__init__()
             self.model = AutoModelForSeq2SeqLM.from_pretrained(LANGUAGE_MODEL_NAME) # Load dummy model
             self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
             if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
             self.model_dim = STAGE5_LM_DIM
             self.model_type = STAGE5_LM_TYPE
             self.use_lora = STAGE5_USE_LORA
         def get_model_dim(self): return self.model_dim
         def prepare_inputs(self, *args, **kwargs): return {'input_ids': torch.randint(0, 100, (2, 10)), 'attention_mask': torch.ones(2, 10)}
         def forward(self, labels=None, **kwargs): return {'loss': torch.tensor(0.0, requires_grad=True) if labels is not None else None} # Dummy output with loss
         def generate(self, *args, **kwargs): return torch.randint(0, 100, (2, 5)) # Dummy generated IDs
         def decode(self, *args, **kwargs): return ["dummy output"] * 2


# --- Configuration ---
LEARNING_RATE = 1e-4 # Adjust based on experiments (might need lower for full model tuning, higher for LoRA)
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3 # Adjust as needed for the full dataset
MODEL_SAVE_PATH = "./neuroreport_model_checkpoint" # Directory to save checkpoints
MAX_LABEL_LENGTH = 256 # Max length for target labels (reports/answers)
WARMUP_STEPS_RATIO = 0.1 # Ratio of total steps for LR warmup
GRADIENT_CLIP_VAL = 1.0 # Optional gradient clipping

# --- Instantiate components (assuming they are loaded/imported correctly) ---
try:
    # Instantiate based on imported classes/configs
    # These should use the actual configurations defined in each stage file if run as a pipeline
    vision_encoder_instance = VisionEncoder()
    slice_aggregator_instance = SliceAggregator(feature_dim=vision_encoder_instance.feature_dim) # Pass the vision dim
    language_decoder_instance = LanguageDecoder() # Uses defaults from stage5 or dummy

    # Determine if bridge is needed based on dimensions
    aggregator_output_dim = slice_aggregator_instance.output_dim
    language_model_dim = language_decoder_instance.get_model_dim()
    if aggregator_output_dim != language_model_dim:
         print(f"Dimensions mismatch: Aggregator ({aggregator_output_dim}) != LM ({language_model_dim}). Using VisionLanguageBridge.")
         bridge_instance = VisionLanguageBridge(visual_dim=aggregator_output_dim, language_dim=language_model_dim)
    else:
         print("Dimensions match. Using Identity bridge.")
         bridge_instance = nn.Identity()

    tokenizer_instance = language_decoder_instance.tokenizer # Get tokenizer from the decoder

except Exception as e_inst:
    print(f"Error instantiating pipeline components for training: {e_inst}")
    print("Check component definitions and configurations in previous stages.")
    # Consider exiting or using fallback defaults if instantiation fails
    exit()


# --- Combined Model (PyTorch Lightning Module) ---
class NeuroReportModel(pl.LightningModule):
    """
    Combines all stages into a single model for end-to-end training using PyTorch Lightning.
    Handles potentially missing annotations during training.
    """
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 slice_aggregator: SliceAggregator,
                 bridge: nn.Module, # Can be VisionLanguageBridge or nn.Identity
                 language_decoder: LanguageDecoder,
                 mode: str = "report", # Default to report as VQA needs questions
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 warmup_steps_ratio: float = WARMUP_STEPS_RATIO,
                 max_label_length: int = MAX_LABEL_LENGTH,
                 **kwargs): # Absorb other potential hparams from config
        super().__init__()
        # Save hyperparameters - important for loading checkpoints
        # Ignore large components to avoid saving them directly in hparams.yaml
        self.save_hyperparameters(ignore=['vision_encoder', 'slice_aggregator', 'bridge', 'language_decoder'])

        self.vision_encoder = vision_encoder
        self.slice_aggregator = slice_aggregator
        self.bridge = bridge # Can be VisionLanguageBridge or nn.Identity
        self.language_decoder = language_decoder # Contains tokenizer and model
        self.mode = mode

        # Store training config params needed for optimizer/scheduler setup
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps_ratio = warmup_steps_ratio # Store ratio
        self.max_label_length = max_label_length
        self.total_training_steps = 10000 # Placeholder, will be updated by trainer
        self.warmup_steps = int(self.total_training_steps * self.warmup_steps_ratio) # Initial estimate


        print("\n--- Trainable Parameters in NeuroReportModel ---")
        self._log_trainable_parameters()


    def _log_trainable_parameters(self):
        total_trainable_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                print(f"  - Trainable: {name} ({param.numel():,})")
                total_trainable_params += param.numel()
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {total_trainable_params:,}")
        if total_params > 0:
             print(f"Trainable Ratio: {total_trainable_params / total_params * 100:.4f}%")
        print("-" * 40 + "\n")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for training/evaluation (calculates loss).

        Args:
            pixel_values (torch.Tensor): Batch of image slices [B, S, C, H, W]
            input_ids (torch.Tensor, optional): Tokenized input text (question/prefix) [B, L_in]
            attention_mask (torch.Tensor, optional): Attention mask for input_ids [B, L_in]
            labels (torch.Tensor, optional): Tokenized target text (answer/report) [B, L_out]. Required for loss.

        Returns:
            Model output containing loss (e.g., Seq2SeqLMOutput or CausalLMOutputWithPast). Loss is None if labels not provided.
        """
        # Stage 2: Vision Encoding
        visual_features_slices = self.vision_encoder(pixel_values) # (B, S, D_v)

        # Stage 3: Slice Aggregation
        aggregated_visual_features = self.slice_aggregator(visual_features_slices) # (B, D_agg)

        # Stage 4: Bridging (Projection or Identity)
        conditioned_embedding = self.bridge(aggregated_visual_features) # (B, D_l)

        # Stage 5: Language Model Processing
        # Prepare inputs for the specific language model type
        model_inputs = {"return_dict": True}

        if self.language_decoder.model_type == 'seq2seq':
            # Reshape visual embedding for encoder_outputs: (B, 1, D_l)
            encoder_hidden_states = conditioned_embedding.unsqueeze(1)
            model_inputs["encoder_outputs"] = (encoder_hidden_states,)
            # Seq2Seq models use input_ids as decoder_input_ids implicitly or explicitly
            # If input_ids are None (e.g., report generation started by decoder_start_token),
            # the generate method handles it, but forward needs something if labels are present.
            # If fine-tuning, input_ids (like prefix) and labels must be provided.
            if input_ids is not None: model_inputs["input_ids"] = input_ids
            if attention_mask is not None: model_inputs["attention_mask"] = attention_mask # Decoder attention mask
            if labels is not None: model_inputs["labels"] = labels

        elif self.language_decoder.model_type == 'causal_lm':
            # Causal LMs typically need `inputs_embeds` for multimodal input.
            # Constructing inputs_embeds requires token embeddings.
            # Simple approach: Pass visual features via cross-attention if model supports it (rare for standard LMs).
            # Workaround: Prepend visual features to text embeddings.
            if input_ids is None: # Should have input_ids (at least BOS) for CausalLM
                 raise ValueError("input_ids are required for CausalLM forward pass during training.")

            # Get text embeddings
            language_embeds = self.language_decoder.model.get_input_embeddings()(input_ids) # [B, L_in, D_l]

            # --- Combine Visual and Text Embeddings ---
            # Prepend the single conditioned visual embedding to the sequence of text embeddings.
            # Visual embedding shape: [B, D_l] -> Unsqueeze to [B, 1, D_l]
            visual_embeds_prep = conditioned_embedding.unsqueeze(1) # [B, 1, D_l]

            # Concatenate: [B, 1, D_l] and [B, L_in, D_l] -> [B, 1 + L_in, D_l]
            inputs_embeds = torch.cat([visual_embeds_prep, language_embeds], dim=1)
            model_inputs["inputs_embeds"] = inputs_embeds

            # --- Prepare corresponding attention mask and labels ---
            # Attention mask needs to account for the added visual token.
            # Visual token mask: [B, 1] (all ones)
            visual_attention_mask = torch.ones(conditioned_embedding.shape[0], 1, dtype=torch.long, device=self.device)
            # Concatenate with text attention mask: [B, 1] and [B, L_in] -> [B, 1 + L_in]
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
            model_inputs["attention_mask"] = combined_attention_mask

            # Labels also need shifting or masking for the visual token.
            # Create labels for the visual token (e.g., -100 to ignore loss)
            visual_labels = torch.full((conditioned_embedding.shape[0], 1), -100, dtype=torch.long, device=self.device)
            # Concatenate with text labels: [B, 1] and [B, L_in] -> [B, 1 + L_in]
            combined_labels = torch.cat([visual_labels, labels], dim=1)
            model_inputs["labels"] = combined_labels

        else:
            raise ValueError(f"Unsupported language model type: {self.language_decoder.model_type}")

        # Check if we have labels to calculate loss
        if "labels" not in model_inputs or model_inputs["labels"] is None:
            # If no labels, run inference pass (e.g., just get logits)
            # Remove 'labels' key if present but None
            model_inputs.pop("labels", None)
            with torch.no_grad(): # No gradients needed if not calculating loss
                 outputs = self.language_decoder.model(**model_inputs)
            outputs.loss = None # Explicitly set loss to None
        else:
             # Run forward pass with labels to get loss
             outputs = self.language_decoder.model(**model_inputs)

        return outputs


    def _prepare_batch(self, batch):
        """Helper to tokenize text and prepare labels for loss calculation, handling missing annotations."""
        pixel_values = batch['pixel_values'].to(self.device)
        batch_size = pixel_values.shape[0]

        input_texts = None
        target_texts = None
        labels = None
        input_ids = None
        attention_mask = None

        # Use tokenizer from the language_decoder instance
        tokenizer = self.language_decoder.tokenizer

        # --- Determine inputs and targets based on mode and available keys ---
        if self.mode == "vqa":
            if 'question' in batch and 'answer' in batch:
                 input_texts = [f"question: {q} context: " for q in batch['question']]
                 target_texts = batch['answer']
            else: # Handle missing VQA annotations
                 warnings.warn("VQA mode selected, but 'question' or 'answer' missing in batch. Cannot train on this batch.")
                 return pixel_values, None, None, None # Return None for texts/labels

        elif self.mode == "report":
            if 'report' in batch: # Use report if available
                 target_texts = batch['report']
                 # Choose prefix based on model type
                 if self.language_decoder.model_type == 'seq2seq':
                      input_texts = ["generate report: "] * batch_size
                 else: # CausalLM - needs starting token(s)
                      input_texts = [tokenizer.bos_token] * batch_size if tokenizer.bos_token else [""] * batch_size
            else: # Handle missing report annotations
                 warnings.warn("Report mode selected, but 'report' missing in batch. Cannot train on this batch.")
                 return pixel_values, None, None, None # Return None for texts/labels
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vqa' or 'report'.")

        # --- Tokenize inputs if available ---
        if input_texts:
            try:
                input_encoding = tokenizer(
                    input_texts, return_tensors='pt', padding='longest', truncation=True, max_length=512
                )
                input_ids = input_encoding.input_ids.to(self.device)
                attention_mask = input_encoding.attention_mask.to(self.device)
            except Exception as e_tok_in:
                warnings.warn(f"Error tokenizing input texts: {e_tok_in}")
                input_ids, attention_mask = None, None # Mark as failed

        # --- Tokenize targets (labels) if available ---
        if target_texts:
            try:
                target_encoding = tokenizer(
                    target_texts, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_label_length
                )
                labels = target_encoding.input_ids.to(self.device)
                # Replace padding token id in labels with -100 for CrossEntropyLoss
                labels[labels == tokenizer.pad_token_id] = -100
            except Exception as e_tok_tgt:
                 warnings.warn(f"Error tokenizing target texts: {e_tok_tgt}")
                 labels = None # Mark as failed

        return pixel_values, input_ids, attention_mask, labels

    def training_step(self, batch, batch_idx):
        prep_result = self._prepare_batch(batch)
        if prep_result is None: # Should not happen with current logic, but good practice
            warnings.warn(f"Skipping training step {batch_idx}: Batch preparation failed fundamentally.")
            return None
        pixel_values, input_ids, attention_mask, labels = prep_result

        # Check if we have labels for this batch (essential for training)
        if labels is None or (self.mode == 'vqa' and input_ids is None):
             # If labels are missing, or VQA inputs are missing, skip training this batch.
             warnings.warn(f"Skipping training step {batch_idx}: Missing required inputs/labels for mode '{self.mode}'.")
             return None # Skip step

        # Perform forward pass to get loss
        try:
            # For CausalLM, labels are prepared inside forward based on inputs_embeds structure
            if self.language_decoder.model_type == 'causal_lm':
                 outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else: # Seq2Seq
                 outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        except Exception as e_fwd:
             warnings.warn(f"Error during training forward pass {batch_idx}: {e_fwd}")
             # Maybe return None or a zero tensor to avoid crashing trainer
             return torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss

        if loss is None:
             warnings.warn(f"Loss is None for training batch {batch_idx}. Check model output and label preparation.")
             return None # Skip if loss calculation failed

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # Remove validation_step as no validation split is used
    # def validation_step(self, batch, batch_idx):
    #     pass # No validation

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
             warnings.warn("No trainable parameters found! Check model freezing logic and LoRA setup.")
             optimizer = optim.AdamW([nn.Parameter(torch.zeros(1))], lr=self.learning_rate) # Dummy
        else:
             print(f"Configuring optimizer for {len(trainable_params)} trainable parameter tensors.")
             optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning Rate Scheduler (requires trainer access for total steps)
        # The scheduler setup depends on when configure_optimizers is called relative to trainer init
        try:
            # Calculate total steps (required for some schedulers)
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                 self.total_training_steps = self.trainer.estimated_stepping_batches
                 print(f"Using trainer's estimated_stepping_batches: {self.total_training_steps}")
            else:
                 # Estimate manually if trainer attribute not available yet
                 # This might happen if called before trainer.fit
                 warnings.warn("Trainer attribute 'estimated_stepping_batches' not found. Estimating total steps.")
                 # Placeholder estimation logic (adjust if needed)
                 self.total_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs // getattr(self.trainer,'accumulate_grad_batches', 1)
                 if self.total_training_steps <= 0: self.total_training_steps = 10000 # Fallback
                 print(f"Manually estimated total steps: {self.total_training_steps}")


            self.warmup_steps = int(self.total_training_steps * self.warmup_steps_ratio)
            print(f"Warmup steps calculated: {self.warmup_steps}")

            scheduler = get_scheduler(
                name="linear", # Example: linear warmup and decay
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step", # Call scheduler step-wise
                "frequency": 1,
            }
            print(f"Configured 'linear' LR scheduler with {self.warmup_steps} warmup steps.")
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        except Exception as e_sched:
             warnings.warn(f"Could not configure LR scheduler: {e_sched}. Using optimizer only.")
             return optimizer


# --- Example Training Setup (PyTorch Lightning) ---
if __name__ == "__main__":
    print("--- Stage 6 Example ---")

    # --- Dummy DataModule Setup ---
    # Use paths appropriate for the BraTS dataset downloaded via KaggleHub
    # Assumes the kagglehub download placed it in './awsaf49-brats2020-training-data'
    # Adjust DATA_DIR if your download path is different
    DATA_DIR_S6 = "./awsaf49-brats2020-training-data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" # Adjusted path
    # Annotations might need to be created or downloaded separately. Using dummy path.
    ANNOTATIONS_PATH_S6 = "./dummy_brats_annotations.csv" # Adjusted dummy path name

    if not os.path.exists(DATA_DIR_S6):
         print(f"ERROR: BraTS data directory not found at '{DATA_DIR_S6}'.")
         print("Please download the dataset using KaggleHub and ensure the path is correct.")
         exit()

    # Recreate dummy annotations if needed, matching patient IDs from the actual data
    if not os.path.exists(ANNOTATIONS_PATH_S6):
         print(f"Creating dummy annotations file: {ANNOTATIONS_PATH_S6}")
         # List actual patient IDs found in the data dir for the example
         actual_patient_ids = [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR_S6, "BraTS20_Training_*"))[:10]] # Use first 10 patients
         if not actual_patient_ids:
              print(f"Warning: Could not find example patient IDs in {DATA_DIR_S6} for dummy annotations.")
              actual_patient_ids = [f"BraTS20_Training_{i:03}" for i in range(1,11)] # Fallback IDs

         dummy_annot_data = []
         for i, p_id in enumerate(actual_patient_ids):
              dummy_annot_data.append({
                  'patient_id': p_id, # Use actual patient IDs from dataset folders
                  'question': f'Dummy Q for {p_id}',
                  'answer': f'Dummy A for {p_id}',
                  'report': f'Dummy report for patient {p_id}. Contains placeholder findings.'
              })
         pd.DataFrame(dummy_annot_data).to_csv(ANNOTATIONS_PATH_S6, index=False)
         print(f"Created dummy annotations for patient IDs: {[d['patient_id'] for d in dummy_annot_data]}")


    data_module_instance = None
    try:
        data_module_instance = NeuroReportDataModule(
            data_dir=DATA_DIR_S6,
            annotations_path=ANNOTATIONS_PATH_S6, # Use None if you don't have annotations yet
            batch_size=BATCH_SIZE,
            mode="report", # Set mode consistent with dummy data/model task
            target_size=TARGET_SIZE,
            num_workers=0,
            n_slices=NUM_SLICES_PER_SCAN
        )
        # Prepare data to ensure datasets are created before trainer needs them
        data_module_instance.prepare_data()
        data_module_instance.setup('fit')
    except Exception as e_dm:
         print(f"Error creating DataModule: {e_dm}")
         data_module_instance = None # Ensure it's None if setup fails

    # --- Instantiate the main model ---
    if data_module_instance:
        try:
            # Use the component instances created earlier
            neuro_report_model_instance = NeuroReportModel(
                vision_encoder=vision_encoder_instance,
                slice_aggregator=slice_aggregator_instance,
                bridge=bridge_instance,
                language_decoder=language_decoder_instance,
                mode="report", # Should match DataModule mode
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                warmup_steps_ratio=WARMUP_STEPS_RATIO,
                max_label_length=MAX_LABEL_LENGTH
            )

            # --- Configure Trainer ---
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            # Modify checkpointing - no validation loss to monitor
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=MODEL_SAVE_PATH,
                filename='neuroreport-epoch={epoch:02d}-step={step}', # Save based on epoch/step
                save_top_k=-1,       # Save all checkpoints or based on epoch/step
                every_n_epochs=1,    # Save every epoch
                save_last=True
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            # Early stopping requires a validation metric, cannot be used without val_loader
            # early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

            # Determine precision based on QLoRA status
            precision_setting = '32-true' # Default
            if torch.cuda.is_available():
                 if language_decoder_instance.use_lora and hasattr(language_decoder_instance.model, 'quantization_config'): # Check if QLoRA active
                      precision_setting = '32-true' # Recommended for stability with bitsandbytes
                      print("Using QLoRA (4-bit): Trainer precision set to 32-true.")
                 elif torch.cuda.is_bf16_supported():
                      precision_setting = 'bf16-mixed'
                      print("Using bfloat16 mixed precision.")
                 else:
                      precision_setting = '16-mixed'
                      print("Using float16 mixed precision.")

            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto",
                precision=precision_setting,
                callbacks=[checkpoint_callback, lr_monitor], # No EarlyStopping
                log_every_n_steps=10,
                gradient_clip_val=GRADIENT_CLIP_VAL,
                # No validation loop specified
                # limit_train_batches=5, # DEBUG: Use fraction of training data
            )

            print(f"\nPyTorch Lightning Trainer configured:")
            print(f"  - Mode: {neuro_report_model_instance.mode}")
            print(f"  - Max Epochs: {NUM_EPOCHS}")
            print(f"  - Precision: {trainer.precision}")
            print(f"  - Checkpoint Path: {MODEL_SAVE_PATH}")
            print("\nStarting training (fit call commented out for example)...")

            # --- Start Training ---
            # No val_dataloaders provided to fit()
            # trainer.fit(neuro_report_model_instance, train_dataloaders=data_module_instance.train_dataloader())
            print("\nTrainer.fit(...) call is commented out.")
            print("To run training, ensure valid DataLoader, dependencies, and uncomment the trainer.fit line.")

        except Exception as e:
            import traceback
            print(f"\nError setting up or running training: {e}")
            traceback.print_exc()
    else:
        print("Skipping Trainer setup because DataModule could not be initialized.")

    print("\nStage 6: Training setup complete.\n")

# Expose key training config for potential use in evaluation/inference stages
# MODEL_SAVE_PATH, MAX_LABEL_LENGTH
```
  </change>
  <change>
    <file>src/pipeline/stage7_evaluation.py</file>
    <description>Remove evaluation logic as requested, since no validation/test split is performed.</description>
    <content><![CDATA[# ========== Stage 7: Evaluation and Validation ==========
import warnings

# --- Evaluation Disabled ---
# As per user request, no validation/test split is performed during training.
# Therefore, this evaluation stage is currently disabled.
# To enable evaluation, modify Stage 1 to create a validation/test split,
# and then implement the evaluation logic here using the test dataloader
# and appropriate metrics (e.g., BLEU, ROUGE, METEOR for reports, Accuracy/F1 for VQA).

warnings.warn("Stage 7 (Evaluation) is disabled as no validation/test split is configured in Stage 1.")

def evaluate_pipeline(*args, **kwargs):
    """Placeholder evaluation function (disabled)."""
    print("Evaluation is disabled.")
    return {}, [], []

# --- Example Usage Placeholder ---
if __name__ == "__main__":
    print("--- Stage 7 Example (Disabled) ---")
    print("Evaluation logic is removed because no validation/test data split was requested.")
    print("Modify Stage 1 and Stage 7 to enable evaluation on a hold-out set.")
    print("\nStage 7: Evaluation setup skipped.\n")

# Define placeholder NeuroReportEvaluator class if it's imported elsewhere
class NeuroReportEvaluator:
     def __init__(self, *args, **kwargs): pass
     def evaluate(self, *args, **kwargs): return {}

# Define placeholder NeuroReportModel class if it's imported elsewhere
try:
     import pytorch_lightning as pl
     class NeuroReportModel(pl.LightningModule): pass
except ImportError:
     class NeuroReportModel: pass # Basic placeholder if pl missing

# Define placeholders for constants if needed
MODEL_SAVE_PATH = "./neuroreport_model_checkpoint_dummy"
MAX_LABEL_LENGTH = 256
BATCH_SIZE = 4
TARGET_SIZE = (224, 224)
NUM_SLICES_PER_SCAN = 64

# Define dummy DataModule/Dataset if needed
try:
     import torch
     from torch.utils.data import DataLoader, Dataset
     import pytorch_lightning as pl
     class MRIDataset(Dataset): pass
     class NeuroReportDataModule(pl.LightningDataModule): pass
except ImportError:
     class MRIDataset: pass
     class NeuroReportDataModule: pass

```
  </change>
  <change>
    <file>src/pipeline/stage9_training_script.py</file>
    <description>Remove validation/test split arguments and logic, disable testing after training, and remove validation-related callbacks (EarlyStopping, checkpoint monitoring val_loss).</description>
    <content><![CDATA[# ========== Stage 9: Training Script ==========
import argparse
import pytorch_lightning as pl
import os
import warnings
import json
import torch
import glob # To find checkpoints

# --- Import necessary components ---
try:
    from .stage1_data_acquisition import NeuroReportDataModule # Data Handling
    from .stage6_training import NeuroReportModel # Model Definition
    # Evaluation components are removed as testing is disabled
    # from .stage7_evaluation import NeuroReportEvaluator, evaluate_pipeline
except ImportError as e:
    warnings.warn(f"Could not import all components from previous stages: {e}. Training script might fail.")
    # Define minimal placeholders only if essential for script structure
    class NeuroReportDataModule(pl.LightningDataModule): pass
    class NeuroReportModel(pl.LightningModule): pass
    # class NeuroReportEvaluator: pass # Not needed
    # def evaluate_pipeline(*args, **kwargs): return {}, [], [] # Not needed


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train the NeuroReport Model")

    # --- Data Arguments ---
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing BraTS patient folders (e.g., ./kagglehub_download/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData)")
    parser.add_argument("--annotations_path", type=str, default=None, help="Path to optional annotations file (.csv or .json) with 'patient_id' column")
    parser.add_argument("--mode", type=str, choices=["vqa", "report"], default="report", help="Task mode ('report' default as VQA needs annotations)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224], help="Target [height, width] for MRI slices")
    parser.add_argument("--n_slices", type=int, default=64, help="Number of slices to select/pad per scan")
    parser.add_argument("--normalization", type=str, default="zero_mean_unit_var", choices=["zero_mean_unit_var", "min_max", "none"], help="Normalization strategy")
    parser.add_argument("--modalities", type=str, nargs='+', default=['t1ce', 'flair'], help="List of MRI modalities to load (e.g., t1ce flair t1 t2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    # Remove val/test split args
    # parser.add_argument("--val_split", type=float, default=0.0, help="Validation split ratio (disabled)")
    # parser.add_argument("--test_split", type=float, default=0.0, help="Test split ratio (disabled)")

    # --- Model Arguments ---
    parser.add_argument("--vision_model_name", type=str, default="vit_base_patch16_224", help="Vision encoder model name (from timm)")
    # parser.add_argument("--freeze_vision", action="store_true", help="Freeze vision backbone") # Add if needed
    parser.add_argument("--aggregation_type", type=str, choices=["lstm", "gru", "transformer", "mean"], default="lstm", help="Slice aggregation method")
    parser.add_argument("--language_model_name", type=str, default="microsoft/BioGPT-Large", help="Language decoder model name (HF)")
    parser.add_argument("--language_model_type", type=str, choices=["causal_lm", "seq2seq"], default=None, help="Override LM type (optional, usually inferred)")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable 4-bit quantization")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True, help="Enable LoRA adapters")

    # --- Training Arguments ---
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum training epochs")
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1, help="Ratio of total steps for LR warmup")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision ('32-true', '16-mixed', 'bf16-mixed')")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator ('cpu', 'gpu', 'tpu', 'auto')")
    parser.add_argument("--devices", default="auto", help="Devices to use (int, list, 'auto')")
    parser.add_argument("--strategy", type=str, default="auto", help="Distributed strategy ('ddp', 'fsdp', 'auto')")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./neuroreport_checkpoints", help="Checkpoint directory")
    # Remove early stopping (needs validation metric)
    # parser.add_argument("--early_stopping_patience", type=int, default=0, help="Patience for early stopping (disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_label_length", type=int, default=256, help="Max sequence length for labels")

    # --- Action Arguments ---
    # Remove testing after train flag
    # parser.add_argument("--run_test_after_train", action=argparse.BooleanOptionalAction, default=False, help="Run evaluation on test set after training (disabled)")

    args = parser.parse_args()

    # --- Post-processing and Validation ---
    # Infer LM type if not specified
    if args.language_model_type is None:
        if "t5" in args.language_model_name.lower() or "bart" in args.language_model_name.lower():
             args.language_model_type = "seq2seq"
        else: args.language_model_type = "causal_lm"
        print(f"Inferred language_model_type: {args.language_model_type}")

    # Validate GPU requirements
    if (args.use_4bit or "16" in args.precision) and not torch.cuda.is_available():
        warnings.warn("CUDA not available. Disabling 4-bit/16-bit precision. Setting precision to 32-true.")
        args.use_4bit = False; args.use_lora = False; args.precision = "32-true"
    if args.use_lora and not args.use_4bit:
        warnings.warn("QLoRA setup requires use_4bit=True. Disabling LoRA."); args.use_lora = False

    args.target_size = tuple(args.target_size) # Ensure tuple for size
    os.makedirs(args.checkpoint_dir, exist_ok=True) # Create checkpoint dir

    # Validate data dir exists
    if not os.path.isdir(args.data_dir):
         raise FileNotFoundError(f"Data directory not found: {args.data_dir}. Please ensure the path is correct and the dataset is downloaded.")
    # Validate annotations path if provided
    if args.annotations_path and not os.path.isfile(args.annotations_path):
         warnings.warn(f"Annotations file not found at {args.annotations_path}. Proceeding without annotations.")
         args.annotations_path = None

    return args

# --- Training Function ---
def train_neuroreport(config):
    """Sets up and runs the PyTorch Lightning training loop."""
    pl.seed_everything(config.seed)

    # 1. DataModule
    print("Initializing DataModule..."); t_start = torch.cuda.Event(enable_timing=True); t_end = torch.cuda.Event(enable_timing=True); t_start.record()
    try:
        data_module = NeuroReportDataModule(
            data_dir=config.data_dir, annotations_path=config.annotations_path,
            batch_size=config.batch_size, mode=config.mode, target_size=config.target_size,
            # val_split=0.0, test_split=0.0, # No splits
            num_workers=config.num_workers,
            normalization=config.normalization, n_slices=config.n_slices,
            modalities=config.modalities
        )
        data_module.prepare_data()
        data_module.setup(stage='fit') # Setup only training data
        t_end.record(); torch.cuda.synchronize(); print(f"DataModule initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing DataModule: {e}"); return None, None

    # Check if dataloader is empty
    if not data_module.train_dataloader():
         print("Error: Training dataloader is empty. Check data directory and dataset setup.")
         return None, None


    # 2. Model
    print("Initializing NeuroReportModel..."); t_start.record()
    try:
        # Model initialization (uses imported classes and config)
        # Warmup steps calculation needs refinement without validation loop
        try:
             # Estimate steps based only on training dataloader
             if hasattr(data_module, 'train_dataloader') and data_module.train_dataloader() is not None:
                  steps_per_epoch = len(data_module.train_dataloader()) // config.accumulate_grad_batches
                  total_steps = steps_per_epoch * config.max_epochs if config.max_epochs > 0 else 10000 # Estimate if max_epochs=-1
                  warmup_steps = int(total_steps * config.warmup_steps_ratio)
                  print(f"Warmup Steps Ratio: {config.warmup_steps_ratio}, Total Est. Steps: {total_steps} -> Warmup Steps: {warmup_steps}")
             else: raise ValueError("Train dataloader not available for step estimation.")
        except Exception as e_steps:
             warmup_steps = 100 # Fallback warmup steps
             print(f"Could not estimate total steps ({e_steps}). Using default warmup_steps: {warmup_steps}")

        # Instantiate the main model class from Stage 6
        model = NeuroReportModel(
            # Pass necessary args based on NeuroReportModel's __init__ signature
            vision_model_name=config.vision_model_name,
            language_model_name=config.language_model_name,
            language_model_type=config.language_model_type,
            aggregation_type=config.aggregation_type,
            use_4bit=config.use_4bit,
            use_lora=config.use_lora,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            mode=config.mode,
            warmup_steps_ratio=config.warmup_steps_ratio, # Pass ratio
            max_label_length=config.max_label_length,
            # Add other args like target_size, n_slices, normalization if needed by model init
        )
        t_end.record(); torch.cuda.synchronize(); print(f"NeuroReportModel initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing NeuroReportModel: {e}"); import traceback; traceback.print_exc(); return None, None

    # 3. Callbacks
    print("Initializing Callbacks..."); t_start.record()
    callbacks = []
    # Checkpointing without validation metric - save based on epoch/step
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"neuroreport-{config.mode}-{{epoch:02d}}-{{step}}", # Include step count
        save_top_k=-1,       # Save all checkpoints or based on interval
        every_n_epochs=1,    # Save checkpoint every epoch
        save_last=True )
    callbacks.append(checkpoint_callback)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    # Remove EarlyStopping as it needs a validation metric
    # if config.early_stopping_patience > 0: ...
    t_end.record(); torch.cuda.synchronize(); print(f"Callbacks initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")

    # 4. Trainer
    print("Initializing Trainer..."); t_start.record()
    # Handle 'auto' device parsing for PL
    devices_param = config.devices
    if isinstance(config.devices, str) and config.devices.lower() == 'auto':
        devices_param = 'auto'
    elif isinstance(config.devices, str): # Handle comma-separated list like "0,1"
         try: devices_param = [int(d.strip()) for d in config.devices.split(',')]
         except ValueError: warnings.warn(f"Invalid device string '{config.devices}'. Using 'auto'.") ; devices_param = 'auto'
    elif isinstance(config.devices, int):
        devices_param = config.devices # Pass integer directly

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator, devices=devices_param,
        strategy=config.strategy if config.strategy != "auto" else None,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val if config.gradient_clip_val > 0 else None,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        # No validation loop: num_sanity_val_steps=0, check_val_every_n_epoch set high? or default PL behavior
        num_sanity_val_steps=0, # Disable sanity check as there's no val loader
        # logger=... # Add logger (TensorBoard, WandB) here if needed
    )

    # Update total steps in model for scheduler *after* trainer init (more reliable)
    if hasattr(trainer, 'estimated_stepping_batches') and trainer.estimated_stepping_batches:
         # This might be None if validation loop is disabled, handle carefully
         model.total_training_steps = trainer.estimated_stepping_batches
         new_warmup = int(model.total_training_steps * config.warmup_steps_ratio)
         if new_warmup != model.warmup_steps:
              print(f"Revising warmup steps based on trainer estimate: {new_warmup}")
              model.warmup_steps = new_warmup
         print(f"Trainer estimated total steps: {model.total_training_steps}")
    else:
         # Fallback to manual estimation if trainer attribute not available
         try:
             train_loader = data_module.train_dataloader()
             model.total_training_steps = len(train_loader) * config.max_epochs // config.accumulate_grad_batches
             if model.total_training_steps <=0: model.total_training_steps = 1000 # Min fallback
             new_warmup = int(model.total_training_steps * config.warmup_steps_ratio)
             print(f"Manually estimated total steps: {model.total_training_steps}. Revised warmup steps: {new_warmup}")
             model.warmup_steps = new_warmup
         except Exception as e_est:
             warnings.warn(f"Could not manually estimate total steps for scheduler ({e_est}). Using previous estimate.")


    t_end.record(); torch.cuda.synchronize(); print(f"Trainer initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")

    # 5. Training
    print("\n--- Starting Training (No Validation Loop) ---"); t_start.record()
    try:
        # Pass only train dataloader
        trainer.fit(model, train_dataloaders=data_module.train_dataloader())
        t_end.record(); torch.cuda.synchronize(); fit_time = t_end.elapsed_time(t_start)/1000
        print(f"--- Training Finished ({fit_time:.2f}s) ---")
    except Exception as e_fit: print(f"Error during training: {e_fit}"); import traceback; traceback.print_exc(); return None, None

    # 6. Testing (Disabled)
    test_results = None
    print("Skipping testing phase as no test split is configured.")
    # if config.run_test_after_train: ... (Removed)

    return trainer, test_results

# --- KaggleHub Dataset Download ---
def download_brats_data(download_dir="./brats2020_kagglehub_download"):
    """Downloads BraTS 2020 dataset using kagglehub."""
    try:
        import kagglehub
        print("Downloading BraTS 2020 dataset from KaggleHub...")
        path = kagglehub.dataset_download(
            "awsaf49/brats2020-training-data",
            path=download_dir, # Specify download location
            force_download=False # Set to True to redownload
        )
        print(f"Dataset downloaded (or already present) at: {path}")
        # Return the expected path to the actual patient data directory
        expected_data_path = os.path.join(path, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
        if os.path.isdir(expected_data_path):
            return expected_data_path
        else:
            warnings.warn(f"Downloaded dataset structure unexpected. Expected patient folders in {expected_data_path}. Please check the download.")
            return path # Return base path if structure is different
    except ImportError:
        print("Error: kagglehub library not found. Please install it: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Error downloading dataset from KaggleHub: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- NeuroReport Training Script ---")
    config = parse_args()

    # Download data if data_dir is not valid or specified via a special flag (optional)
    # Simple check: If config.data_dir is not a directory, try downloading.
    if not os.path.isdir(config.data_dir):
         print(f"Data directory '{config.data_dir}' not found.")
         download_path = download_brats_data() # Download to default ./brats2020_kagglehub_download
         if download_path:
             config.data_dir = download_path # Update config to use downloaded path
             print(f"Updated data directory to downloaded path: {config.data_dir}")
         else:
             print("Dataset download failed. Exiting.")
             exit()


    print("\n--- Configuration ---")
    for k, v in vars(config).items(): print(f"  {k}: {v}")
    print("-" * 21 + "\n")

    # Run Training
    training_output = train_neuroreport(config)

    if training_output and training_output[0] is not None: # Check if trainer object exists
        print("\nTraining script completed successfully.")
        trainer = training_output[0]
        # Optionally save the final model explicitly if needed beyond checkpointing
        # final_save_path = os.path.join(config.checkpoint_dir, "neuroreport_final.ckpt")
        # trainer.save_checkpoint(final_save_path)
        # print(f"Final model checkpoint saved to: {final_save_path}")

        # Find the last checkpoint to mention it
        last_ckpt_path = os.path.join(config.checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_path):
             print(f"Last checkpoint saved at: {last_ckpt_path}")
        else: # Find latest checkpoint by modification time if last.ckpt missing
             ckpt_files = glob.glob(os.path.join(config.checkpoint_dir, "*.ckpt"))
             if ckpt_files:
                 latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                 print(f"Latest checkpoint saved at: {latest_ckpt}")

    else:
        print("\nTraining script failed or was interrupted.")

```
  </change>
  <change>
    <file>src/pipeline/stage10_main.py</file>
    <description>Update main entry point to reflect removal of evaluation/demo flags and dataset splitting. Focus on running the training script.</description>
    <content><![CDATA[# ========== Stage 10: Main Entry Point ==========
import argparse
import os
import warnings
import json
import torch
import pytorch_lightning as pl

# --- Import pipeline components ---
try:
    # Training script function and its parser
    from .stage9_training_script import train_neuroreport, parse_args, download_brats_data
    # Evaluation/Demo components are removed/disabled
    # from .stage7_evaluation import evaluate_pipeline, NeuroReportEvaluator # Disabled
    # from .stage8_inference import launch_demo # Disabled
    # Model and DataModule classes might be needed if loading checkpoint outside training script
    from .stage6_training import NeuroReportModel
    from .stage1_data_acquisition import NeuroReportDataModule
except ImportError as e:
    warnings.warn(f"Error importing pipeline components: {e}. Some functionalities might not be available.")
    # Define dummy functions/classes if needed for script structure
    def parse_args(): return argparse.Namespace(mode='report', data_dir=None) # Basic dummy
    def train_neuroreport(config): print("Dummy train function"); return None, None
    def download_brats_data(): print("Dummy download function"); return None
    # def evaluate_pipeline(*args, **kwargs): print("Dummy evaluate function"); return {}, [], [] # Disabled
    # def launch_demo(model_path): print(f"Dummy launch demo function for path: {model_path}") # Disabled
    class NeuroReportModel(pl.LightningModule): pass
    class NeuroReportDataModule(pl.LightningDataModule): pass
    # class NeuroReportEvaluator: pass # Disabled


# --- Main Function ---
def main():
    # Use the unified argument parser from the training script (stage9)
    config = parse_args() # config holds all parsed arguments

    print("--- NeuroReport Pipeline ---")
    print(f"Selected Mode: {config.mode}")

    # --- Download Data if necessary ---
    if not os.path.isdir(config.data_dir):
        print(f"Data directory '{config.data_dir}' not found or invalid.")
        downloaded_path = download_brats_data() # Attempt download
        if downloaded_path and os.path.isdir(downloaded_path):
            config.data_dir = downloaded_path
            print(f"Dataset downloaded/found at: {config.data_dir}")
        else:
            print(f"Error: Could not find or download dataset. Please check the path or run download manually.")
            return # Exit if data is not available

    # --- Action: Run Training ---
    # Since evaluation/demo are removed, the main action is training.
    print("\nInitiating training process...")
    trainer, _ = train_neuroreport(config) # Call the training function

    if trainer and trainer.checkpoint_callback.best_model_path:
         print(f"Training completed. Best checkpoint saved at: {trainer.checkpoint_callback.best_model_path}")
    elif trainer and trainer.checkpoint_callback.last_model_path:
         print(f"Training completed. Last checkpoint saved at: {trainer.checkpoint_callback.last_model_path}")
    else:
         print("Training did not complete successfully or no checkpoints were saved.")


    # --- Post-Training Actions (Removed) ---
    # Evaluation (--evaluate flag) - Removed
    # Demo (--demo flag) - Removed

    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    # Set environment variables if needed before execution
    # e.g., os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()

```
  </change>
  <change>
    <file>README.md</file>
    <description>Update README with instructions for downloading data and running the training pipeline.</description>
    <content><![CDATA[# NeuroReport: Medical Imaging VQA & Report Generation Pipeline

This project implements a pipeline for Visual Question Answering (VQA) and report generation on medical MRI scans, specifically designed for the BraTS 2020 dataset.

## Overview

The pipeline consists of several stages:

1.  **Data Acquisition & Preprocessing**: Loads BraTS 2020 NIfTI files, handles multiple modalities (e.g., T1ce, FLAIR), preprocesses slices (normalization, resizing), and prepares data batches. Uses `kagglehub` to download the dataset.
2.  **Vision Encoder**: Extracts features from individual MRI slices using a Vision Transformer (ViT) model.
3.  **Slice Aggregation**: Combines slice features into a single scan-level embedding using methods like LSTM, GRU, Transformer, or mean pooling.
4.  **Vision-Language Bridge**: Projects the visual embedding to match the language model's expected dimension, enabling conditioning. Uses Cross-Attention.
5.  **Language Decoder**: Generates text (reports or VQA answers) using a biomedical language model (e.g., BioGPT, Flan-T5), conditioned on the visual features. Supports QLoRA for efficient fine-tuning.
6.  **Training Module**: Integrates all components into a PyTorch Lightning module for end-to-end training.
7.  **Evaluation**: (Currently Disabled) Placeholder for evaluating model performance using metrics like BLEU, ROUGE, METEOR, Accuracy.
8.  **Inference Demo**: (Currently Disabled) Placeholder for an interactive Gradio demo.
9.  **Training Script**: Main script to configure and run the training process.
10. **Main Entry Point**: Orchestrates the pipeline execution (currently focuses on training).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up Kaggle API Token:**
    *   Go to your Kaggle account settings (`https://www.kaggle.com/<your-username>/account`).
    *   Click "Create New API Token". This will download `kaggle.json`.
    *   Place the `kaggle.json` file in the expected location (usually `~/.kaggle/kaggle.json`).
    *   Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install torch torchvision torchaudio pytorch-lightning timm transformers peft bitsandbytes accelerate sentencepiece pandas scikit-learn nibabel opencv-python evaluate rouge_score sacrebleu nltk gradio kagglehub
    # If using METEOR, download NLTK data:
    # python -m nltk.downloader wordnet omw-1.4 punkt
    ```
    *Note: `bitsandbytes` might require specific build steps depending on your CUDA version.*

## Usage

The pipeline is primarily executed through `src/pipeline/stage10_main.py`, which uses `src/pipeline/stage9_training_script.py` to handle training.

1.  **Download Dataset (Handled Automatically):**
    The training script (`stage9`) will attempt to download the `awsaf49/brats2020-training-data` dataset using `kagglehub` if the specified `--data_dir` is not found. It will download to `./brats2020_kagglehub_download` by default. Ensure your Kaggle API token is set up.

2.  **Prepare Annotations (Optional but Recommended):**
    *   Create a CSV or JSON file containing annotations.
    *   It **must** have a `patient_id` column that matches the BraTS folder names (e.g., `BraTS20_Training_001`).
    *   For **report generation** (`--mode report`), include a `report` column with the target text.
    *   For **VQA** (`--mode vqa`), include `question` and `answer` columns.
    *   If no annotations file is provided (`--annotations_path None`), the model trains solely on the images (unsupervised pre-training or generation without explicit targets). The script will create a dummy annotations file for structure if none is found at the default path.

3.  **Run Training:**
    Execute the main script, providing necessary arguments.

    **Example (Report Generation Mode, using defaults):**
    ```bash
    python -m src.pipeline.stage10_main --data_dir ./brats2020_kagglehub_download/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --mode report --annotations_path ./path/to/your/report_annotations.csv --max_epochs 5 --batch_size 2 --accelerator gpu --devices 1 --checkpoint_dir ./brats_report_checkpoints
    ```

    **Example (VQA Mode, custom LM, QLoRA disabled):**
    ```bash
    python -m src.pipeline.stage10_main --data_dir ./brats2020_kagglehub_download/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --mode vqa --annotations_path ./path/to/your/vqa_annotations.csv --language_model_name google/flan-t5-base --use_4bit=false --use_lora=false --max_epochs 10 --batch_size 4 --learning_rate 5e-5 --checkpoint_dir ./brats_vqa_checkpoints
    ```

    **Key Arguments:**
    *   `--data_dir`: Path to the directory containing `BraTS20_Training_XXX` folders.
    *   `--annotations_path`: Path to your CSV/JSON annotations file (optional).
    *   `--mode`: `vqa` or `report`.
    *   `--max_epochs`: Number of training epochs.
    *   `--batch_size`: Adjust based on GPU memory.
    *   `--accelerator`, `--devices`: Configure hardware usage.
    *   `--checkpoint_dir`: Where to save model checkpoints.
    *   `--use_4bit`, `--use_lora`: Enable/disable QLoRA.
    *   See `src/pipeline/stage9_training_script.py` for all available arguments.

4.  **Inference/Evaluation (Currently Disabled):**
    The evaluation (Stage 7) and demo (Stage 8) components are disabled because the current configuration trains on the entire dataset without a validation/test split. To enable these:
    *   Modify `src/pipeline/stage1_data_acquisition.py` to implement data splitting.
    *   Re-enable and adapt the code in `src/pipeline/stage7_evaluation.py` and `src/pipeline/stage8_inference.py`.
    *   Adjust `src/pipeline/stage9_training_script.py` and `src/pipeline/stage10_main.py` to include evaluation/demo steps and arguments.

## Checkpoints

Model checkpoints will be saved in the directory specified by `--checkpoint_dir` during training. The best/last checkpoint can be used for continued training or inference (once enabled).
```
  </change>
  <change>
    <file>requirements.txt</file>
    <description>Add kagglehub to requirements.</description>
    <content><![CDATA[torch
torchvision
torchaudio
pytorch-lightning
timm
transformers
peft
bitsandbytes
accelerate
sentencepiece
pandas
scikit-learn
nibabel
opencv-python-headless
evaluate
rouge_score
sacrebleu
nltk
gradio
kagglehub
psutil
matplotlib
Pillow
kaggle
    