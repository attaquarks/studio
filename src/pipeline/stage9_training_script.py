# ========== Stage 9: Training Script ==========
import argparse
import pytorch_lightning as pl
import os
import warnings
import json
import torch

# --- Import necessary components ---
# Use try-except for robustness if run standalone vs as part of pipeline execution
# These imports assume the pipeline structure is correct
try:
    from .stage1_data_acquisition import NeuroReportDataModule # Data Handling
    from .stage6_training import NeuroReportModel # Model Definition
    # Import evaluator for testing
    from .stage7_evaluation import NeuroReportEvaluator, evaluate_pipeline # Evaluation logic
except ImportError as e:
    warnings.warn(f"Could not import all components from previous stages: {e}. Training script might fail.")
    # Define minimal placeholders only if essential for script structure, but training will likely fail
    class NeuroReportDataModule(pl.LightningDataModule): pass
    class NeuroReportModel(pl.LightningModule): pass
    class NeuroReportEvaluator: pass
    def evaluate_pipeline(*args, **kwargs): return {}, [], []


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train the NeuroReport Model")

    # --- Data Arguments (from Stage 1 DataModule) ---
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing MRI volumes (.nii/.nii.gz)")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to annotations file (.csv or .json)")
    parser.add_argument("--mode", type=str, choices=["vqa", "report"], default="vqa", help="Task mode: 'vqa' or 'report' generation")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224], help="Target [height, width] for MRI slices")
    parser.add_argument("--n_slices", type=int, default=64, help="Number of slices to select/pad per scan")
    parser.add_argument("--normalization", type=str, default="zero_mean_unit_var", choices=["zero_mean_unit_var", "min_max", "none"], help="Normalization strategy")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")

    # --- Model Arguments (from Stage 6 Model Init) ---
    parser.add_argument("--vision_model_name", type=str, default="vit_base_patch16_224", help="Vision encoder model name (from timm)")
    # parser.add_argument("--freeze_vision", action="store_true", help="Freeze vision backbone") # Add if needed
    parser.add_argument("--aggregation_type", type=str, choices=["lstm", "gru", "transformer", "mean"], default="lstm", help="Slice aggregation method")
    # Add other aggregator/bridge params if they are configurable in NeuroReportModel.__init__
    parser.add_argument("--language_model_name", type=str, default="microsoft/BioGPT-Large", help="Language decoder model name (HF)")
    parser.add_argument("--language_model_type", type=str, choices=["causal_lm", "seq2seq"], default=None, help="Override LM type (optional, usually inferred)")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable 4-bit quantization")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True, help="Enable LoRA adapters")
    # Add LoRA config args if needed

    # --- Training Arguments (for Trainer and Model) ---
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
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_label_length", type=int, default=256, help="Max sequence length for labels")

    # --- Action Arguments ---
    parser.add_argument("--run_test_after_train", action=argparse.BooleanOptionalAction, default=True, help="Run evaluation on test set after training")
    # Add flags if needed to force re-training even if checkpoint exists
    # parser.add_argument("--force_train", action="store_true", help="Force training even if checkpoint exists")

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
            val_split=config.val_split, test_split=config.test_split, num_workers=config.num_workers,
            normalization=config.normalization, n_slices=config.n_slices
        )
        data_module.prepare_data()
        data_module.setup(stage='fit')
        t_end.record(); torch.cuda.synchronize(); print(f"DataModule initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing DataModule: {e}"); return None, None

    # 2. Model
    print("Initializing NeuroReportModel..."); t_start.record()
    try:
        # Calculate warmup steps based on estimated total steps
        # Need to estimate steps per epoch carefully
        try:
             steps_per_epoch = len(data_module.train_dataloader()) // config.accumulate_grad_batches
             total_steps = steps_per_epoch * config.max_epochs if config.max_epochs > 0 else 10000 # Estimate if max_epochs=-1
             warmup_steps = int(total_steps * config.warmup_steps_ratio)
             print(f"Warmup Steps Ratio: {config.warmup_steps_ratio}, Total Est. Steps: {total_steps} -> Warmup Steps: {warmup_steps}")
        except Exception as e_steps:
             warmup_steps = 100 # Fallback warmup steps
             print(f"Could not estimate total steps ({e_steps}). Using default warmup_steps: {warmup_steps}")


        model = NeuroReportModel(
            vision_model_name=config.vision_model_name,
            language_model_name=config.language_model_name,
            language_model_type=config.language_model_type,
            aggregation_type=config.aggregation_type,
            use_4bit=config.use_4bit, use_lora=config.use_lora,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            mode=config.mode, warmup_steps=warmup_steps, # Pass calculated steps
            max_label_length=config.max_label_length,
            # Pass other hparams from config if NeuroReportModel init expects them
            # e.g., target_size=config.target_size, n_slices=config.n_slices
        )
        t_end.record(); torch.cuda.synchronize(); print(f"NeuroReportModel initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")
    except Exception as e: print(f"Error initializing NeuroReportModel: {e}"); import traceback; traceback.print_exc(); return None, None

    # 3. Callbacks
    print("Initializing Callbacks..."); t_start.record()
    callbacks = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", dirpath=config.checkpoint_dir,
        filename=f"neuroreport-{config.mode}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1, mode="min", save_last=True )
    callbacks.append(checkpoint_callback)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    if config.early_stopping_patience > 0:
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, mode="min", verbose=True)
        callbacks.append(early_stopping)
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
        # logger=... # Add logger (TensorBoard, WandB) here if needed
    )
    # Update total steps in model for scheduler *after* trainer init (more reliable)
    if hasattr(trainer, 'estimated_stepping_batches'):
         model.total_training_steps = trainer.estimated_stepping_batches
         new_warmup = int(model.total_training_steps * config.warmup_steps_ratio)
         if new_warmup != model.warmup_steps:
              print(f"Revising warmup steps based on trainer estimate: {new_warmup}")
              model.warmup_steps = new_warmup
         print(f"Trainer estimated total steps: {model.total_training_steps}")
    else:
        warnings.warn("Trainer has no 'estimated_stepping_batches' attribute. Scheduler might use estimated total steps.")


    t_end.record(); torch.cuda.synchronize(); print(f"Trainer initialized ({t_end.elapsed_time(t_start)/1000:.2f}s).")

    # 5. Training
    print("\n--- Starting Training ---"); t_start.record()
    try:
        trainer.fit(model, datamodule=data_module)
        t_end.record(); torch.cuda.synchronize(); fit_time = t_end.elapsed_time(t_start)/1000
        print(f"--- Training Finished ({fit_time:.2f}s) ---")
    except Exception as e_fit: print(f"Error during training: {e_fit}"); import traceback; traceback.print_exc(); return None, None

    # 6. Testing (Optional)
    test_results = None
    if config.run_test_after_train:
        print("\n--- Starting Testing ---"); t_start.record()
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
             print(f"Loading best model from: {best_ckpt_path}")
             # Setup test dataloader
             try: data_module.setup(stage='test')
             except Exception as e_dm_test: print(f"Error setting up test data: {e_dm_test}"); best_ckpt_path=None # Prevent testing

             if best_ckpt_path: # Check again if setup succeeded
                 test_trainer = pl.Trainer(accelerator=config.accelerator, devices=devices_param, precision=config.precision, logger=False) # New trainer for testing
                 test_results = test_trainer.test(model=model, datamodule=data_module, ckpt_path=best_ckpt_path)
                 t_end.record(); torch.cuda.synchronize(); test_time = t_end.elapsed_time(t_start)/1000
                 print(f"--- Testing Finished ({test_time:.2f}s) ---")
                 if test_results: print("Test Results:", json.dumps(test_results, indent=2))
        else: print("Best checkpoint not found. Skipping testing.")
    else: print("Skipping testing phase.")

    return trainer, test_results

# --- Main Execution ---
if __name__ == "__main__":
    print("--- NeuroReport Training Script ---")
    config = parse_args()
    print("\n--- Configuration ---")
    for k, v in vars(config).items(): print(f"  {k}: {v}")
    print("-" * 21 + "\n")

    # Run Training
    training_output = train_neuroreport(config)

    if training_output and training_output[0] is not None: # Check if trainer object exists
        print("\nTraining script completed successfully.")
        # trainer, test_results = training_output
    else:
        print("\nTraining script failed or was interrupted.")
