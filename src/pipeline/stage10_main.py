# ========== Stage 10: Main Entry Point ==========
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
    <file>README.md</file