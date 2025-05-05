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
    # Re-enable evaluation components
    from .stage7_evaluation import evaluate_pipeline # Function to run eval
    from .stage8_inference import launch_demo # Demo function
    from .stage6_training import NeuroReportModel # Model class for loading checkpoint
    from .stage1_data_acquisition import NeuroReportDataModule # DataModule for eval/demo
    from transformers import AutoTokenizer # Needed for evaluation/demo
except ImportError as e:
    warnings.warn(f"Error importing pipeline components: {e}. Some functionalities might not be available.")
    # Define dummy functions/classes if needed for script structure
    def parse_args(): return argparse.Namespace(mode='report', data_dir=None, checkpoint_path=None, evaluate=False, demo=False) # Basic dummy
    def train_neuroreport(config): print("Dummy train function"); return None, None
    def download_brats_data(): print("Dummy download function"); return None
    def evaluate_pipeline(*args, **kwargs): print("Dummy evaluate function"); return {}, [], [] # Re-enabled dummy
    def launch_demo(model_path): print(f"Dummy launch demo function for path: {model_path}") # Re-enabled dummy
    class NeuroReportModel(pl.LightningModule): pass
    class NeuroReportDataModule(pl.LightningDataModule): pass
    class AutoTokenizer: @staticmethod def from_pretrained(name): return None # Dummy tokenizer


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

    # --- Determine Checkpoint Path ---
    checkpoint_path = config.checkpoint_path # Path specified by user
    trainer = None # Initialize trainer

    # --- Action: Run Training (if no checkpoint specified) ---
    if checkpoint_path is None:
        print("\nNo checkpoint path provided. Initiating training process...")
        training_output = train_neuroreport(config) # Call the training function
        if training_output and training_output[0] is not None:
             trainer = training_output[0] # Get trainer object
             test_results = training_output[1] # Get test results if run_test_after_train was True
             # Get best checkpoint path from the callback after training
             if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
                  checkpoint_path = trainer.checkpoint_callback.best_model_path
                  print(f"Training completed. Best checkpoint saved at: {checkpoint_path}")
             elif trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'last_model_path') and trainer.checkpoint_callback.last_model_path:
                  checkpoint_path = trainer.checkpoint_callback.last_model_path
                  print(f"Training completed. Using last checkpoint: {checkpoint_path}")
             else:
                  print("Training finished, but could not find the best or last checkpoint path.")
                  checkpoint_path = None # Ensure it's None if path couldn't be found
        else:
             print("Training did not complete successfully or was interrupted.")
             return # Exit if training failed and no checkpoint was provided

    # --- Action: Run Evaluation (if specified and checkpoint available) ---
    if config.evaluate:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n--- Running Evaluation on Checkpoint: {checkpoint_path} ---")
            try:
                # Load model from checkpoint
                model = NeuroReportModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
                tokenizer = model.tokenizer # Get tokenizer from loaded model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # Setup DataModule for testing
                eval_data_module = NeuroReportDataModule(
                    data_dir=config.data_dir,
                    annotations_path=config.annotations_path,
                    batch_size=config.batch_size, # Use training batch size or add eval_batch_size arg
                    mode=config.mode,
                    target_size=config.target_size,
                    val_split=0.0, # Don't need val split
                    test_split=1.0, # Use all data as test for evaluation in this context
                    num_workers=config.num_workers,
                    n_slices=config.n_slices,
                    modalities=config.modalities,
                    normalization=config.normalization,
                )
                eval_data_module.prepare_data()
                eval_data_module.setup('test')
                test_loader = eval_data_module.test_dataloader()

                if test_loader:
                    # Run evaluation using the function from stage7
                    results, _, _ = evaluate_pipeline(
                        model=model,
                        dataloader=test_loader,
                        tokenizer=tokenizer,
                        device=device,
                        is_encoder_decoder=model.is_encoder_decoder_model,
                        max_gen_length=config.max_label_length, # Use label length as estimate
                        num_beams=4 # Example beams
                    )
                    print("\nEvaluation Results:")
                    print(json.dumps(results, indent=2))
                else:
                    print("Could not create test dataloader for evaluation.")

            except Exception as e_eval:
                print(f"Error during evaluation: {e_eval}")
                import traceback; traceback.print_exc()
        else:
            print("\nSkipping evaluation: Checkpoint path is invalid or not provided.")


    # --- Action: Launch Demo (if specified and checkpoint available) ---
    if config.demo:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n--- Launching Inference Demo with Checkpoint: {checkpoint_path} ---")
            try:
                launch_demo(model_path=checkpoint_path)
            except Exception as e_demo:
                 print(f"Error launching demo: {e_demo}")
        else:
             print("\nSkipping demo: Checkpoint path is invalid or not provided.")


    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    # Set environment variables if needed before execution
    # e.g., os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
