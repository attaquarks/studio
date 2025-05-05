# ========== Stage 7: Evaluation and Validation ==========
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from evaluate import load as load_metric # HuggingFace Evaluate library
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score # Using custom logic for exact match VQA
import warnings
import numpy as np
import os
import glob

# Assume a trained model checkpoint exists or use the model from Stage 6
# Assume an evaluation DataLoader (`test_loader`) exists for the test set
# Assume the tokenizer is available

# --- Import Components ---
try:
    from .stage6_training import NeuroReportModel, MODEL_SAVE_PATH # Import model and default save path
    from .stage1_data_acquisition import NeuroReportDataModule # Import DataModule for test loader setup
    # Need tokenizer info, often part of the model or loaded separately
except ImportError:
     warnings.warn("Could not import necessary components from previous stages. Evaluation might fail.")
     class NeuroReportModel(pl.LightningModule): pass
     class NeuroReportDataModule(pl.LightningDataModule): pass
     MODEL_SAVE_PATH = "./dummy_checkpoints"


# --- Configuration ---
# Path to the trained model checkpoint (if loading from file)
DEFAULT_CHECKPOINT_PATH = None # Example: f"{MODEL_SAVE_PATH}/neuroreport-epoch=01-val_loss=1.23.ckpt" # Replace with actual path or let auto-find
EVAL_BATCH_SIZE = 8 # Can be larger than training batch size for inference
MAX_GEN_LENGTH_EVAL = 128 # Max length for generation during evaluation
NUM_BEAMS_EVAL = 4 # Beam search for evaluation generation

# --- Load Metrics ---
print("Loading evaluation metrics...")
try:
    # VQA Metrics (if applicable) - Handled manually via exact match below

    # Report Generation Metrics
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")
    meteor_metric = load_metric("meteor")
    print("Metrics loaded: BLEU, ROUGE, METEOR.")
except Exception as e:
    print(f"Warning: Failed to load some metrics (ensure 'evaluate', 'nltk', 'rouge_score', 'sacrebleu' are installed): {e}")
    bleu_metric, rouge_metric, meteor_metric = None, None, None # Set to None if loading failed

# --- Evaluation Function ---
@torch.no_grad() # Disable gradients for inference
def evaluate_pipeline(model, dataloader, tokenizer, device, is_encoder_decoder, max_gen_length, num_beams):
    """
    Evaluates the full NeuroReport model on a given dataloader.

    Args:
        model (pl.LightningModule): The trained NeuroReportModel instance.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        tokenizer: The tokenizer used for the language model.
        device: The device to run evaluation on ('cuda' or 'cpu').
        is_encoder_decoder (bool): Flag indicating if the language model is Seq2Seq.
        max_gen_length (int): Max tokens for generated output.
        num_beams (int): Number of beams for generation.

    Returns:
        tuple: (dict: calculated metrics, list: all predictions, list: all references)
    """
    model.eval() # Set model to evaluation mode
    model.to(device)

    all_predictions = []
    all_references = []
    is_vqa_task = None # Determine task type from the first batch

    if dataloader is None:
         print("Error: Evaluation dataloader is None. Cannot perform evaluation.")
         return {}, [], []

    print(f"Starting evaluation on {len(dataloader.dataset)} samples...")
    batch_num = 0
    for batch in dataloader:
        batch_num += 1
        print(f"  Processing batch {batch_num}/{len(dataloader)}...")
        pixel_values = batch['pixel_values'].to(device)

        # Determine task type on first batch
        if is_vqa_task is None:
            is_vqa_task = 'question' in batch
            print(f"Detected task type: {'VQA' if is_vqa_task else 'Report Generation'}")

        # Prepare ground truth references
        if is_vqa_task:
            if 'question' not in batch or 'answer' not in batch:
                warnings.warn(f"Skipping batch {batch_num}: Missing 'question' or 'answer' key for VQA.")
                continue
            references = batch['answer']
            # Prepare input prompts for generation (match training format)
            input_texts = [f"question: {q} context: " for q in batch['question']]
        else: # Report Generation
            if 'report' not in batch:
                 warnings.warn(f"Skipping batch {batch_num}: Missing 'report' key for Report Generation.")
                 continue
            references = batch['report']
            # Prepare input prompts (match training format)
            if is_encoder_decoder:
                 input_texts = ["generate report: "] * len(references)
            else: # CausalLM
                 input_texts = [tokenizer.bos_token] * len(references) if tokenizer.bos_token else [""] * len(references)


        all_references.extend(references)

        # --- Perform Inference using the combined model ---
        # (Replicates forward logic for generation)

        # 1. Vision Encoding (handled within model.forward for training loss, but need it here for generation)
        slice_features = model.vision_encoder(pixel_values)
        # 2. Aggregation
        scan_embedding = model.slice_aggregator(slice_features)
        # 3. Bridge
        conditioned_embedding = model.bridge(scan_embedding)
        # 4. Prepare LM inputs
        encoder_hidden_states = conditioned_embedding.unsqueeze(1) # (B, 1, Dim)

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # 5. Generate using the language model component
        lm_to_generate = model.language_decoder.model # Access the underlying HF model
        generation_kwargs = {
            "max_length": max_gen_length + input_ids.shape[1], # max_length includes prompt length for generate
            "num_beams": num_beams,
            "early_stopping": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if is_encoder_decoder:
             generation_kwargs["encoder_outputs"] = (encoder_hidden_states,)
             generation_kwargs["input_ids"] = input_ids # Use tokenized prompt
             # generation_kwargs["attention_mask"] = attention_mask # Generate usually handles this based on input_ids
        else: # CausalLM
             generation_kwargs["input_ids"] = input_ids
             generation_kwargs["attention_mask"] = attention_mask
             # Standard CausalLM might ignore encoder_outputs. Needs multimodal arch or inputs_embeds.
             warnings.warn("Evaluation generation for CausalLM uses basic input; visual conditioning might be limited.")

        try:
             output_sequences = lm_to_generate.generate(**generation_kwargs)
             # Decode predictions, removing prompt part if necessary
             # For Seq2Seq, generate output doesn't include the prompt
             # For CausalLM, generate output *does* include the prompt, so slice it off
             if not is_encoder_decoder:
                 predictions = tokenizer.batch_decode(output_sequences[:, input_ids.shape[1]:], skip_special_tokens=True)
             else:
                 predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
             all_predictions.extend(predictions)

        except Exception as e_gen:
             print(f"Error during generation for batch {batch_num}: {e_gen}")
             # Add dummy predictions to maintain alignment with references
             all_predictions.extend(["<GENERATION_ERROR>"] * len(references))

    print("Generation complete. Calculating metrics...")
    # --- Calculate Metrics ---
    results = {}
    if not all_predictions or len(all_predictions) != len(all_references):
         print(f"Warning: Mismatch in prediction ({len(all_predictions)}) and reference ({len(all_references)}) counts. Cannot calculate metrics accurately.")
         return results, all_predictions, all_references

    if is_vqa_task:
        print("Calculating VQA Metrics (Exact Match Accuracy)...")
        # Simple exact match accuracy
        exact_match = [1 if pred.strip().lower() == ref.strip().lower() else 0 for pred, ref in zip(all_predictions, all_references)]
        accuracy = np.mean(exact_match) if exact_match else 0.0
        results['vqa_accuracy_exact_match'] = accuracy
        print(f"  Accuracy (Exact Match): {accuracy:.4f}")
        # F1/Precision/Recall would require token-level comparison logic here

    else: # Report Generation Metrics
        print("Calculating Report Generation Metrics (BLEU, ROUGE, METEOR)...")
        # Filter out generation errors before calculating metrics
        valid_indices = [i for i, pred in enumerate(all_predictions) if pred != "<GENERATION_ERROR>"]
        valid_predictions = [all_predictions[i] for i in valid_indices]
        valid_references = [all_references[i] for i in valid_indices]

        if not valid_predictions:
            print("No valid predictions available to calculate text generation metrics.")
            return results, all_predictions, all_references

        try:
             if bleu_metric:
                 # BLEU requires tokenized inputs (list of lists of strings)
                 bleu_preds = [pred.split() for pred in valid_predictions]
                 bleu_refs = [[ref.split()] for ref in valid_references] # List of list of lists format
                 bleu_score = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)
                 results['bleu'] = bleu_score['bleu'] # Extract main BLEU score
                 print(f"  BLEU: {results.get('bleu', 0.0):.4f}")
             else: print("  BLEU metric not loaded, skipping.")
        except Exception as e: print(f"Could not calculate BLEU: {e}")

        try:
            if rouge_metric:
                 rouge_score = rouge_metric.compute(predictions=valid_predictions, references=valid_references)
                 # Store individual ROUGE scores
                 results.update({k: v for k, v in rouge_score.items()}) # Adds rouge1, rouge2, rougeL, rougeLsum
                 print(f"  ROUGE-1: {results.get('rouge1', 0.0):.4f}, ROUGE-2: {results.get('rouge2', 0.0):.4f}, ROUGE-L: {results.get('rougeL', 0.0):.4f}")
            else: print("  ROUGE metric not loaded, skipping.")
        except Exception as e: print(f"Could not calculate ROUGE: {e}")

        try:
             if meteor_metric:
                 meteor_score = meteor_metric.compute(predictions=valid_predictions, references=valid_references)
                 results['meteor'] = meteor_score['meteor']
                 print(f"  METEOR: {results.get('meteor', 0.0):.4f}")
             else: print("  METEOR metric not loaded, skipping.")
        except ImportError: # Catch NLTK download errors specifically for METEOR
             print("  METEOR calculation skipped: NLTK data potentially missing (try `python -m nltk.downloader wordnet omw-1.4 punkt`).")
        except Exception as e: print(f"Could not calculate METEOR: {e}")


    print("Evaluation Metrics Calculation Complete.")
    return results, all_predictions, all_references


# --- Example Evaluation Usage ---
if __name__ == "__main__":
    print("--- Stage 7 Example ---")

    # --- Setup: Find Checkpoint and Load Model ---
    ckpt_path_to_eval = DEFAULT_CHECKPOINT_PATH
    # Auto-find best checkpoint if default path is None
    if ckpt_path_to_eval is None:
        try:
             if os.path.isdir(MODEL_SAVE_PATH):
                  ckpt_files = glob.glob(os.path.join(MODEL_SAVE_PATH, "neuroreport-*.ckpt"))
                  # Filter out 'last.ckpt' and find the one with the best 'val_loss' in filename
                  best_ckpt = None
                  best_loss = float('inf')
                  for ckpt in ckpt_files:
                       if 'last' in os.path.basename(ckpt).lower(): continue
                       try:
                           # Example filename: neuroreport-epoch=02-val_loss=1.23.ckpt
                           loss_str = ckpt.split('val_loss=')[1].split('.ckpt')[0]
                           loss = float(loss_str)
                           if loss < best_loss:
                                best_loss = loss
                                best_ckpt = ckpt
                       except (IndexError, ValueError): continue # Ignore files not matching pattern

                  if best_ckpt:
                       ckpt_path_to_eval = best_ckpt
                       print(f"Auto-found best checkpoint: {ckpt_path_to_eval} (val_loss={best_loss:.4f})")
                  else: # Fallback to last.ckpt if no best found
                       last_ckpt = os.path.join(MODEL_SAVE_PATH, 'last.ckpt')
                       if os.path.exists(last_ckpt):
                            ckpt_path_to_eval = last_ckpt
                            print(f"Using last checkpoint: {ckpt_path_to_eval}")
             else: print(f"Checkpoint directory not found: {MODEL_SAVE_PATH}")
        except Exception as e: print(f"Error auto-finding checkpoint: {e}")

    neuro_report_model = None
    tokenizer = None
    if ckpt_path_to_eval and os.path.exists(ckpt_path_to_eval):
         print(f"Attempting to load model from checkpoint: {ckpt_path_to_eval}")
         try:
              # Loading requires the class definition and potentially dependencies
              neuro_report_model = NeuroReportModel.load_from_checkpoint(ckpt_path_to_eval, map_location='cpu') # Load to CPU first
              tokenizer = neuro_report_model.tokenizer # Get tokenizer from loaded model
              print("Model and tokenizer loaded successfully.")
         except Exception as e:
              print(f"Failed to load model from checkpoint: {e}")
              import traceback; traceback.print_exc()
    else:
        print("No valid checkpoint path found or specified. Cannot run evaluation.")

    # --- Setup: Get Test Dataloader ---
    test_loader = None
    if neuro_report_model: # Only setup dataloader if model loaded
        try:
            # Need data dir and annotations path - potentially from model hparams or config
            data_dir = getattr(neuro_report_model.hparams, 'data_dir', './BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData') # Example fallback
            annotations_path = getattr(neuro_report_model.hparams, 'annotations_path', './dummy_brats_annotations.csv')
            mode = getattr(neuro_report_model.hparams, 'mode', 'report')
            target_size = getattr(neuro_report_model.hparams, 'target_size', (224, 224))
            n_slices = getattr(neuro_report_model.hparams, 'n_slices', 64)
            modalities = getattr(neuro_report_model.hparams, 'modalities', ['t1ce', 'flair'])
            normalization = getattr(neuro_report_model.hparams, 'normalization', 'zero_mean_unit_var')


            # Assuming NeuroReportDataModule is available
            eval_data_module = NeuroReportDataModule(
                data_dir=data_dir,
                annotations_path=annotations_path,
                batch_size=EVAL_BATCH_SIZE, # Use eval batch size
                mode=mode,
                target_size=target_size,
                val_split=0.0, # Don't need val split for testing
                test_split=1.0, # Use all provided data as test data for evaluation
                num_workers=0, # Often simpler for eval
                n_slices=n_slices,
                modalities=modalities,
                normalization=normalization
            )
            eval_data_module.prepare_data()
            eval_data_module.setup('test') # Setup the test dataset
            test_loader = eval_data_module.test_dataloader()
            if not test_loader: print("Warning: Test dataloader setup failed.")

        except Exception as e_data:
             print(f"Error setting up DataModule for evaluation: {e_data}")


    # --- Run Evaluation ---
    if neuro_report_model and test_loader and tokenizer:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            print(f"\nRunning evaluation on {eval_device}...")
            eval_results, predictions, references = evaluate_pipeline(
                model=neuro_report_model,
                dataloader=test_loader,
                tokenizer=tokenizer,
                device=eval_device,
                is_encoder_decoder=neuro_report_model.is_encoder_decoder_model, # Get from model instance
                max_gen_length=MAX_GEN_LENGTH_EVAL,
                num_beams=NUM_BEAMS_EVAL
            )

            print("\n--- Evaluation Results ---")
            if eval_results:
                 for metric, value in eval_results.items(): print(f"{metric}: {value:.4f}")
            else: print("No metrics calculated.")

            # Print some examples
            print("\n--- Sample Predictions vs References ---")
            num_samples_to_show = 5
            if predictions and references:
                 for i in range(min(num_samples_to_show, len(predictions))):
                      print(f"Sample {i+1}:")
                      print(f"  Reference: {references[i]}")
                      print(f"  Prediction: {predictions[i]}\n")
            else: print("No predictions/references to show.")

        except Exception as e_eval:
            import traceback
            print(f"Error during evaluation run: {e_eval}")
            traceback.print_exc()
    else:
        print("\nSkipping evaluation run because the model, dataloader, or tokenizer is not available.")

    print("\nStage 7: Evaluation setup complete.\n")
