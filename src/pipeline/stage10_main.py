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
    from .stage9_training_script import train_neuroreport, parse_args
    # Evaluation function (if separate evaluation is desired)
    from .stage7_evaluation import evaluate_pipeline, NeuroReportEvaluator
    # Inference demo function
    from .stage8_inference import launch_demo
    # Model and DataModule classes needed for loading checkpoints/data
    from .stage6_training import NeuroReportModel
    from .stage1_data_acquisition import NeuroReportDataModule
except ImportError as e:
    warnings.warn(f"Error importing pipeline components: {e}. Some functionalities might not be available.")
    # Define dummy functions/classes if needed for script structure
    def parse_args(): return argparse.Namespace(evaluate=False, demo=False, checkpoint_path=None) # Basic dummy
    def train_neuroreport(config): print("Dummy train function"); return None, None
    def evaluate_pipeline(*args, **kwargs): print("Dummy evaluate function"); return {}, [], []
    def launch_demo(model_path): print(f"Dummy launch demo function for path: {model_path}")
    class NeuroReportModel(pl.LightningModule): pass
    class NeuroReportDataModule(pl.LightningDataModule): pass
    class NeuroReportEvaluator: pass


# --- Main Function ---
def main():
    # Use the unified argument parser from the training script (stage9)
    config = parse_args() # config holds all parsed arguments

    print("--- NeuroReport Pipeline ---")
    print(f"Selected Mode: {config.mode}")

    # --- Determine Action ---
    # Primarily driven by flags like --evaluate, --demo, and presence of --checkpoint_path

    best_checkpoint_path = None # Will store path to best model after training/loading

    # Check if only evaluation or demo is requested with a specific checkpoint
    # If --evaluate or --demo is True AND a checkpoint_path is given, skip training.
    should_skip_train = (config.evaluate or config.demo) and config.checkpoint_path and os.path.exists(config.checkpoint_path)

    if should_skip_train:
        print(f"\nCheckpoint provided ({config.checkpoint_path}) and evaluation/demo requested. Skipping training.")
        best_checkpoint_path = config.checkpoint_path
    else:
        # Default action: Train the model (or load checkpoint if --checkpoint_path is given without --evaluate/--demo)
        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
             print(f"\nCheckpoint provided ({config.checkpoint_path}), but neither --evaluate nor --demo specified. Will use this checkpoint for subsequent actions if any, otherwise exiting.")
             # Optionally add a flag like --continue_training to resume from checkpoint
             best_checkpoint_path = config.checkpoint_path
             # If no further actions, the script will just exit after this message.
        else:
             print("\nNo valid checkpoint provided or resuming/eval/demo not specified. Initiating training process...")
             trainer, test_results_train = train_neuroreport(config) # Call the training function
             if trainer:
                 best_checkpoint_path = trainer.checkpoint_callback.best_model_path
                 print(f"Training completed. Best checkpoint saved at: {best_checkpoint_path}")
             else:
                 print("Training failed. Exiting.")
                 return # Exit if training failed

    # --- Post-Training / Standalone Actions ---

    # Evaluation (--evaluate flag)
    # This block runs if --evaluate is true, using the best_checkpoint_path found/provided.
    if config.evaluate:
        print("\n--- Running Standalone Evaluation ---")
        if not best_checkpoint_path or not os.path.exists(best_checkpoint_path):
             print(f"Error: Cannot evaluate. Checkpoint path not found or invalid: {best_checkpoint_path}")
        else:
             try:
                  print(f"Loading model from checkpoint: {best_checkpoint_path}")
                  model_to_eval = NeuroReportModel.load_from_checkpoint(best_checkpoint_path, map_location='cpu')
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  model_to_eval.to(device)
                  model_to_eval.eval()

                  print("Setting up DataModule for evaluation...")
                  # Use config args to setup DataModule for the test set
                  eval_data_module = NeuroReportDataModule(
                        data_dir=config.data_dir, annotations_path=config.annotations_path,
                        batch_size=config.batch_size * 2, mode=model_to_eval.mode, # Use loaded model's mode
                        target_size=config.target_size, val_split=0.0, test_split=config.test_split if config.test_split > 0 else 0.1,
                        num_workers=config.num_workers, normalization=config.normalization, n_slices=config.n_slices
                  )
                  eval_data_module.prepare_data()
                  eval_data_module.setup(stage='test')
                  test_loader = eval_data_module.test_dataloader()

                  if not test_loader or len(test_loader.dataset) == 0:
                       print("Warning: Test dataset is empty. Cannot run evaluation.")
                  else:
                       print(f"Starting evaluation on {len(test_loader.dataset)} test samples...")
                       # Call the evaluation function from Stage 7
                       eval_results, _, _ = evaluate_pipeline(
                            model=model_to_eval, dataloader=test_loader, device=device,
                            # max_gen_length=config.max_label_length # Pass necessary params
                       )
                       print("\n--- Evaluation Results ---")
                       print(json.dumps(eval_results, indent=2))

             except FileNotFoundError: print(f"Error: Checkpoint file not found at {best_checkpoint_path}.")
             except Exception as e_eval: print(f"Error during evaluation: {e_eval}"); import traceback; traceback.print_exc()

    # Demo (--demo flag)
    # This block runs if --demo is true, using the best_checkpoint_path found/provided.
    if config.demo:
        print("\n--- Launching Gradio Demo ---")
        if not best_checkpoint_path or not os.path.exists(best_checkpoint_path):
             print(f"Error: Cannot launch demo. Checkpoint path not found or invalid: {best_checkpoint_path}")
        else:
             try:
                  # Call the demo launch function from Stage 8
                  launch_demo(model_path=best_checkpoint_path)
             except Exception as e_demo: print(f"Error launching Gradio demo: {e_demo}"); import traceback; traceback.print_exc()

    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    # Set environment variables if needed before execution
    # e.g., os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
