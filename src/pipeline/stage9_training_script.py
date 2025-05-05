# ========== Stage 9: Training Script ==========
import argparse
import pytorch_lightning as pl
import os
import warnings
import json
import torch
import glob # To find checkpoints
import pandas as pd # For creating dummy annotations

# --- Import necessary components ---
try:
    from .stage1_data_acquisition import NeuroReportDataModule # Data Handling
    from .stage6_training import NeuroReportModel # Model Definition
    # Evaluation components (re-enable if needed)
    from .stage7_evaluation import evaluate_pipeline, NeuroReportEvaluator
except ImportError as e:
    warnings.warn(f"Could not import all components from previous stages: {e}. Training script might fail.")
    # Define minimal placeholders only if essential for script structure
    class NeuroReportDataModule(pl.LightningDataModule): pass
    class NeuroReportModel(pl.LightningModule): pass
    class NeuroReportEvaluator: pass
    def evaluate_pipeline(*args, **kwargs): return {}, [], []


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
    # Re-enable val/test split args
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")

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
    # Re-enable early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping based on val_loss (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_label_length", type=int, default=256, help="Max sequence length for labels")

    # --- Action Arguments ---
    # Re-enable testing after train flag
    parser.add_argument("--run_test_after_train", action=argparse.BooleanOptionalAction, default=True, help="Run evaluation on test set after training")

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
    # Validate annotations path if provided, create dummy if None
    if args.annotations_path is None:
         args.annotations_path = "./dummy_brats_annotations_training.csv"
         if not os.path.exists(args.annotations_path):
              print(f"No annotations path provided. Creating dummy annotations at {args.annotations_path}")
              # Create dummy annotations based on data_dir
              try:
                   patient_ids = [os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, "BraTS20_Training_*")) if os.path.isdir(p)]
                   if not patient_ids: raise FileNotFoundError("No patient folders found for dummy annotations.")
                   dummy_data = [{'patient_id': pid, 'question': f'Dummy Q {pid}', 'answer': f'Dummy A {pid}', 'report': f'Dummy R {pid}'} for pid in patient_ids]
                   pd.DataFrame(dummy_data).to_csv(args.annotations_path, index=False)
                   print(f"Created dummy annotations for {len(patient_ids)} patients.")
              except Exception as e: print(f"Failed to create dummy annotations: {e}"); exit()
         else: print(f"Using existing dummy annotations at {args.annotations_path}")
    elif not os.path.isfile(args.annotations_path):
         raise FileNotFoundError(f"Annotations file not found at {args.annotations_path}. Please provide a valid path or omit the argument to create a dummy file.")

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
            val_split=config.val_split, test_split=config.test_split, # Use splits from config
            num_workers=config.num_workers,
            normalization=config.normalization, n_slices=config.n_slices,
            modalities=config.modalities
        )
        data_module.prepare_data()
        data_module.setup(stage='fit') # Setup train/val splits
        t_end.record(); torch.cuda.synchronize(); print(f"DataModule initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing DataModule: {e}"); return None, None

    # Check if dataloader is empty
    if not data_module.train_dataloader():
         print("Error: Training dataloader is empty. Check data directory and dataset setup.")
         return None, None
    if config.val_split > 0 and not data_module.val_dataloader():
         print("Warning: Validation split > 0 but validation dataloader is empty.")


    # 2. Model
    print("Initializing NeuroReportModel..."); t_start.record()
    try:
        # Get tokenizer from DataModule (assuming it's loaded/created there or in Stage 5)
        # In this setup, model needs tokenizer. We load it here or assume it's part of LanguageDecoder.
        # Simplified: Re-load tokenizer here based on config (Ideally pass from Stage 5)
        from transformers import AutoTokenizer
        tokenizer_instance = AutoTokenizer.from_pretrained(config.language_model_name)
        if tokenizer_instance.pad_token is None: tokenizer_instance.pad_token = tokenizer_instance.eos_token

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
            tokenizer=tokenizer_instance, # Pass tokenizer to model
            is_encoder_decoder_model=(config.language_model_type == 'seq2seq'), # Pass type flag
            # Add other args like target_size, n_slices, normalization if needed by model init
        )
        t_end.record(); torch.cuda.synchronize(); print(f"NeuroReportModel initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing NeuroReportModel: {e}"); import traceback; traceback.print_exc(); return None, None

    # 3. Callbacks
    print("Initializing Callbacks..."); t_start.record()
    callbacks = []
    # Re-enable checkpointing monitoring val_loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"neuroreport-{config.mode}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,        # Save only the best model
        monitor='val_loss',  # Monitor validation loss
        mode='min',          # Minimize validation loss
        save_last=True )
    callbacks.append(checkpoint_callback)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    # Re-enable EarlyStopping if patience > 0
    if config.early_stopping_patience > 0:
         early_stopping_callback = pl.callbacks.EarlyStopping(
             monitor='val_loss',
             patience=config.early_stopping_patience,
             mode='min'
         )
         callbacks.append(early_stopping_callback)
         print(f"Enabled EarlyStopping with patience {config.early_stopping_patience} on val_loss.")
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
        # Enable validation loop checking
        check_val_every_n_epoch=1, # Check validation every epoch
        num_sanity_val_steps=2, # Run sanity checks on val data
        # logger=... # Add logger (TensorBoard, WandB) here if needed
    )

    # Update total steps in model for scheduler *after* trainer init (more reliable)
    # Needs adjustment since estimated_stepping_batches includes validation batches
    try:
         if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'train_dataloader') and trainer.datamodule.train_dataloader() is not None:
             num_train_batches = len(trainer.datamodule.train_dataloader())
             accum_factor = getattr(trainer, 'accumulate_grad_batches', 1)
             model.total_training_steps = (num_train_batches // accum_factor) * trainer.max_epochs
             if model.total_training_steps <= 0: model.total_training_steps = 10000 # Fallback
             new_warmup = int(model.total_training_steps * config.warmup_steps_ratio)
             print(f"Estimated train steps per epoch: {num_train_batches // accum_factor}")
             print(f"Total estimated training steps: {model.total_training_steps}. Revised warmup steps: {new_warmup}")
             model.warmup_steps = new_warmup
         else: raise ValueError("Train dataloader not available for step estimation.")
    except Exception as e_est:
         warnings.warn(f"Could not manually estimate total steps for scheduler ({e_est}). Using previous estimate.")


    t_end.record(); torch.cuda.synchronize(); print(f"Trainer initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")

    # 5. Training
    print("\n--- Starting Training ---"); t_start.record()
    try:
        # Pass datamodule to fit for train and val loaders
        trainer.fit(model, datamodule=data_module)
        t_end.record(); torch.cuda.synchronize(); fit_time = t_end.elapsed_time(t_start)/1000
        print(f"--- Training Finished ({fit_time:.2f}s) ---")
    except Exception as e_fit: print(f"Error during training: {e_fit}"); import traceback; traceback.print_exc(); return None, None

    # 6. Testing (Re-enabled)
    test_results = None
    if config.run_test_after_train:
         print("\n--- Starting Testing ---"); t_start.record()
         # Setup test data
         data_module.setup(stage='test')
         if data_module.test_dataloader():
              test_results = trainer.test(model, datamodule=data_module)
              t_end.record(); torch.cuda.synchronize(); test_time = t_end.elapsed_time(t_start)/1000
              print(f"--- Testing Finished ({test_time:.2f}s) ---")
              print("Test Results:", test_results)
         else:
              print("Skipping testing: No test dataloader found.")
    else:
         print("Skipping testing as --run_test_after_train is not set.")


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

        # Find the best checkpoint to mention it
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print(f"Best checkpoint (based on val_loss) saved at: {best_ckpt_path}")
        elif trainer.checkpoint_callback.last_model_path and os.path.exists(trainer.checkpoint_callback.last_model_path):
            print(f"Last checkpoint saved at: {trainer.checkpoint_callback.last_model_path}")
        else:
             print("No checkpoint path found in callback.")

    else:
        print("\nTraining script failed or was interrupted.")
