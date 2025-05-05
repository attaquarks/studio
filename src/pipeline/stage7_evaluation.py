# ========== Stage 7: Evaluation and Validation ==========
import torch
from torch.utils.data import DataLoader
from evaluate import load as load_metric # HuggingFace Evaluate library
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import numpy as np
import os
import pytorch_lightning as pl # For loading checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # For dummy model/tokenizer

# --- Import Components from Previous Stages ---
# Attempt to import necessary classes and variables
# Use try-except blocks to handle cases where stages are run independently
pipeline_dir = os.path.dirname(__file__) # Get directory of the current file

# Need access to the main model class definition and potential dataloader
# And constants like MODEL_SAVE_PATH from stage 6
try:
    from .stage6_training import NeuroReportModel, MODEL_SAVE_PATH, MAX_GENERATION_LENGTH as MAX_LABEL_LENGTH
except ImportError:
    warnings.warn("Could not import NeuroReportModel/constants from stage6. Evaluation might fail if checkpoint loading is needed or definitions differ.")
    # Define placeholder if loading from checkpoint is not the primary goal of this standalone run
    MODEL_SAVE_PATH = "./neuroreport_model_checkpoint_dummy"
    MAX_LABEL_LENGTH = 128 # Must match what model was trained with
    # If NeuroReportModel cannot be imported, loading from checkpoint won't work easily.
    # We might need to define a dummy class or rely on the model instance being passed.
    class NeuroReportModel(pl.LightningModule): # Dummy definition
        def __init__(self, *args, **kwargs):
            super().__init__()
            # Need dummy components if we were to try and run this standalone without a loaded model
            self.vision_encoder = nn.Identity()
            self.slice_aggregator = nn.Identity()
            self.bridge = nn.Identity()
            self.language_model = None # Cannot easily create dummy LM here
            self.tokenizer = None
            self.is_encoder_decoder_model = True # Assume T5 default
        def forward(self, *args, **kwargs): pass # Dummy forward

# Need access to a dataloader from stage 1 (or a dummy)
try:
    from .stage1_data_acquisition import MRIDataset, BATCH_SIZE, NUM_SLICES_PER_SCAN, IMG_SIZE
except ImportError:
    warnings.warn("Could not import from stage1. Defining dummy MRIDataset class and constants for Stage 7.")
    BATCH_SIZE = 2
    NUM_SLICES_PER_SCAN = 8
    IMG_SIZE = 32 # Small dummy size
    # Define a minimal dummy Dataset class
    class MRIDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths, labels, **kwargs):
            self.file_paths = file_paths
            self.labels = labels
            self.num_slices = NUM_SLICES_PER_SCAN
            self.img_size = IMG_SIZE
        def __len__(self): return len(self.file_paths)
        def __getitem__(self, idx):
            label = self.labels[idx]
            dummy_pixels = torch.randn(self.num_slices, 3, self.img_size, self.img_size)
            item = {'pixel_values': dummy_pixels}
            if isinstance(label, dict):
                item['question'] = label.get('question', 'Dummy Q')
                item['answer'] = label.get('answer', 'Dummy A')
            else:
                item['report'] = str(label)
            return item

# Need tokenizer and model from stage 5/6 (if not loading checkpoint)
try:
    from .stage5_language_decoder import tokenizer as loaded_tokenizer, language_model as loaded_language_model, is_encoder_decoder
    if loaded_tokenizer is None or loaded_language_model is None: raise ImportError
    tokenizer = loaded_tokenizer
    language_model = loaded_language_model # Potentially PEFT wrapped
    print("Imported tokenizer and language_model instance from Stage 5.")
except ImportError:
    warnings.warn("Could not import tokenizer/model instance from stage5. Evaluation will require loading from checkpoint or passing instance.")
    tokenizer = None
    language_model = None
    is_encoder_decoder = True # Assume default if model not loaded


# --- Configuration ---
# Path to the trained model checkpoint (if loading from file)
# Construct path using MODEL_SAVE_PATH from stage 6 if needed
# Example: Find the best checkpoint in the save directory
checkpoint_dir = MODEL_SAVE_PATH
best_checkpoint_path = None
if os.path.isdir(checkpoint_dir):
     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and 'last' not in f]
     if checkpoints:
          # Simple heuristic: find based on name pattern (e.g., lowest val_loss)
          checkpoints.sort() # Or sort based on metric in filename
          best_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0]) # Assume lowest loss is first if sorted well
          print(f"Found potential best checkpoint: {best_checkpoint_path}")
     else:
         print(f"No non-last checkpoints found in {checkpoint_dir}.")
         # Check for last.ckpt as fallback
         last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
         if os.path.exists(last_ckpt):
              best_checkpoint_path = last_ckpt
              print(f"Using last checkpoint as fallback: {last_ckpt}")


CHECKPOINT_PATH = best_checkpoint_path # Set to found path or None
EVAL_BATCH_SIZE = 8 # Can be larger than training batch size for inference
MAX_GEN_LENGTH_EVAL = MAX_LABEL_LENGTH # Use same max length as training labels by default
NUM_BEAMS_EVAL = 4 # Beam search for evaluation generation

# --- Load Metrics ---
print("Loading evaluation metrics...")
bleu_metric, rouge_metric, meteor_metric = None, None, None
try:
    bleu_metric = load_metric("bleu")
    print("Loaded BLEU.")
except Exception as e:
    print(f"Warning: Failed to load BLEU metric ('sacrebleu' might be needed): {e}")
try:
    rouge_metric = load_metric("rouge")
    print("Loaded ROUGE.")
except Exception as e:
    print(f"Warning: Failed to load ROUGE metric ('rouge_score' might be needed): {e}")
try:
    meteor_metric = load_metric("meteor")
    print("Loaded METEOR.")
except Exception as e:
    print(f"Warning: Failed to load METEOR metric ('nltk' data might be needed): {e}")


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
    if not hasattr(model, 'vision_encoder') or \
       not hasattr(model, 'slice_aggregator') or \
       not hasattr(model, 'bridge') or \
       not hasattr(model, 'language_model'):
        raise AttributeError("Model instance is missing required components (vision_encoder, slice_aggregator, bridge, language_model). Ensure the correct model object is passed.")

    if model.language_model is None or tokenizer is None:
         raise ValueError("Language model or tokenizer is None within the model object or globally. Cannot evaluate.")

    model.eval() # Set model to evaluation mode
    model.to(device)

    all_predictions = []
    all_references = []
    is_vqa_task = None # Determine task type from the first batch

    print(f"Starting evaluation on {len(dataloader.dataset)} samples using device {device}...")
    for i, batch in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}...")
        pixel_values = batch['pixel_values'].to(device)

        # Determine task type on first batch
        if is_vqa_task is None:
            is_vqa_task = 'question' in batch and 'answer' in batch
            print(f"Detected task type: {'VQA' if is_vqa_task else 'Report Generation'}")

        # Prepare ground truth references
        if is_vqa_task:
            references = batch['answer']
            # Prepare input prompts for generation (match training format)
            # Example: Retrieve prompt format used during training if possible, or use default
            input_texts = [f"question: {q} context: " for q in batch['question']]
        elif 'report' in batch:
            references = batch['report']
            # Prepare input prompts (match training format)
            input_texts = ["generate report: "] * len(references)
        else:
             warnings.warn(f"Batch {i} missing expected keys ('question'/'answer' or 'report'). Skipping batch.")
             continue

        # Filter out empty references if necessary, although metrics handle them
        current_batch_refs = [ref for ref in references if isinstance(ref, str) and len(ref.strip()) > 0]
        if len(current_batch_refs) != len(references):
             warnings.warn(f"Found {len(references) - len(current_batch_refs)} empty/invalid references in batch {i}.")
             # Decide how to handle this - skip batch, or only eval valid pairs?
             # For now, we'll let metrics handle empty refs if possible, but log them.
             # We keep the original `references` list aligned with `input_texts`.

        all_references.extend(references) # Keep original alignment

        # --- Perform Inference using the combined model ---
        # Get visual features -> Aggregate -> Bridge -> Condition LM -> Generate

        # 1. Vision Encoding, Aggregation, Bridge (using the combined model structure)
        # Note: Direct access to intermediate steps is cleaner here than full forward
        try:
            slice_features = model.vision_encoder(pixel_values)
            scan_embedding = model.slice_aggregator(slice_features)
            conditioned_embedding = model.bridge(scan_embedding) # (B, LangModelDim)
        except Exception as e_vis:
             print(f"Error during visual processing stages for batch {i}: {e_vis}. Skipping batch.")
             continue # Skip to next batch

        # 4. Prepare LM inputs (visual part)
        # Ensure correct shape for encoder_hidden_states if needed by LM
        encoder_hidden_states = None
        if is_encoder_decoder:
            encoder_hidden_states = conditioned_embedding.unsqueeze(1) # (B, 1, Dim)

        # 5. Prepare LM inputs (text part - prompt)
        if not input_texts: # Should not happen with current logic but check
             warnings.warn(f"Input texts are empty for batch {i}. Cannot generate. Skipping.")
             continue
        try:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        except Exception as e_tok:
            print(f"Error tokenizing input texts for batch {i}: {e_tok}. Skipping batch.")
            continue

        # 6. Generate using the language model component
        # Access the underlying language model (might be wrapped in PEFT)
        lm_to_generate = model.language_model
        generation_kwargs = {
            "max_length": max_gen_length,
            "num_beams": num_beams,
            "early_stopping": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Prepare inputs for generate method based on model type
        if is_encoder_decoder:
             # Pass visual features as encoder output
             generation_kwargs["encoder_outputs"] = (encoder_hidden_states,) # Needs to be a tuple
             # Pass input_ids and attention_mask (which act as decoder inputs)
             generation_kwargs["input_ids"] = input_ids
             generation_kwargs["attention_mask"] = attention_mask
        else: # CausalLM
             # Basic approach: use input_ids (e.g., BOS or prompt)
             generation_kwargs["input_ids"] = input_ids
             generation_kwargs["attention_mask"] = attention_mask
             # Note: True visual conditioning usually requires inputs_embeds or specific model arch
             # We do NOT pass encoder_outputs here for standard CausalLMs
             warnings.warn("Evaluation generation for CausalLM uses basic text input; visual conditioning might be limited unless using a multimodal architecture.", RuntimeWarning)

        # Run generation
        try:
            output_sequences = lm_to_generate.generate(**generation_kwargs)
        except Exception as e_gen:
             print(f"Error during model.generate() for batch {i}: {e_gen}. Skipping batch.")
             # Add dummy predictions to maintain alignment if needed, or filter later
             all_predictions.extend(["GENERATION_ERROR"] * len(references))
             continue # Skip to next batch


        # Decode predictions
        try:
            # Handle prompt removal for CausalLM if prompt was included in input_ids
            prompt_len_to_remove = 0
            if not is_encoder_decoder and input_ids.shape[1] > 0:
                 # Check if generated output starts with input_ids (common case)
                 if torch.equal(output_sequences[:, :input_ids.shape[1]], input_ids):
                      prompt_len_to_remove = input_ids.shape[1]

            predictions = tokenizer.batch_decode(output_sequences[:, prompt_len_to_remove:], skip_special_tokens=True)
            all_predictions.extend(predictions)
        except Exception as e_dec:
            print(f"Error decoding predictions for batch {i}: {e_dec}. Adding placeholder.")
            all_predictions.extend(["DECODING_ERROR"] * len(references))


    print("Generation complete. Calculating metrics...")
    # --- Calculate Metrics ---
    # Filter out pairs where prediction failed before metric calculation
    valid_indices = [i for i, p in enumerate(all_predictions) if p not in ["GENERATION_ERROR", "DECODING_ERROR"]]
    filtered_predictions = [all_predictions[i] for i in valid_indices]
    filtered_references = [all_references[i] for i in valid_indices]

    results = {}
    if not filtered_predictions:
         print("Warning: No valid predictions available after filtering errors. Cannot calculate metrics.")
         return results, all_predictions, all_references # Return raw lists

    if is_vqa_task is None: # If dataloader was empty
         print("Warning: Task type could not be determined (empty dataloader?). Cannot calculate metrics.")
         return results, all_predictions, all_references

    if is_vqa_task:
        print("Calculating VQA Metrics (Exact Match Accuracy)...")
        # Simple exact match accuracy
        exact_match = [1 if pred.strip().lower() == ref.strip().lower() else 0 for pred, ref in zip(filtered_predictions, filtered_references)]
        accuracy = np.mean(exact_match) if exact_match else 0.0
        results['vqa_accuracy_exact_match'] = accuracy
        print(f"  Accuracy (Exact Match): {accuracy:.4f}")
        # F1/Precision/Recall would need more sophisticated NLP comparison (e.g., token overlap, embedding similarity)
        # Example placeholder using sklearn (treat each unique answer as a class - poor for free text)
        # try:
        #     results['vqa_f1_weighted'] = f1_score(filtered_references, filtered_predictions, average='weighted', zero_division=0) ... etc
        # except Exception as e: print(f"Could not calculate F1/Prec/Recall: {e}")

    else: # Report Generation Metrics
        print("Calculating Report Generation Metrics (BLEU, ROUGE, METEOR)...")
        if bleu_metric:
            try:
                bleu_preds = [pred.split() for pred in filtered_predictions]
                bleu_refs = [[ref.split()] for ref in filtered_references]
                bleu_score = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)
                results['bleu'] = bleu_score['bleu'] if bleu_score else 0.0
                print(f"  BLEU: {results.get('bleu', 0.0):.4f}")
            except Exception as e:
                print(f"Could not calculate BLEU: {e}")
        if rouge_metric:
            try:
                rouge_score = rouge_metric.compute(predictions=filtered_predictions, references=filtered_references)
                results.update({k: v for k, v in rouge_score.items()}) # Adds rouge1, rouge2, rougeL, rougeLsum
                print(f"  ROUGE-1: {results.get('rouge1', 0.0):.4f}, ROUGE-L: {results.get('rougeL', 0.0):.4f}")
            except Exception as e:
                print(f"Could not calculate ROUGE: {e}")
        if meteor_metric:
            try:
                meteor_score = meteor_metric.compute(predictions=filtered_predictions, references=filtered_references)
                results['meteor'] = meteor_score['meteor'] if meteor_score else 0.0
                print(f"  METEOR: {results.get('meteor', 0.0):.4f}")
            except Exception as e:
                print(f"Could not calculate METEOR (ensure NLTK data is downloaded): {e}")


    print("Evaluation Metrics Calculation Complete.")
    # Return raw predictions/references along with filtered metrics
    return results, all_predictions, all_references


# --- Example Evaluation Usage ---
print("--- Stage 7 Example ---")

# --- Setup: Load Model and Data ---
eval_model = None
eval_tokenizer = None
eval_loader = None
model_to_eval_is_encoder_decoder = True # Set default, will be updated if model loads

# Option 1: Use model instance from previous stage (if running sequentially)
if 'neuro_report_model' in locals() and isinstance(neuro_report_model, pl.LightningModule):
    print("Using model instance 'neuro_report_model' from previous stage.")
    eval_model = neuro_report_model
    eval_tokenizer = neuro_report_model.tokenizer # Get tokenizer from model
    model_to_eval_is_encoder_decoder = neuro_report_model.is_encoder_decoder_model
    if eval_tokenizer is None: print("Warning: Tokenizer not found in model object.")
# Option 2: Load from checkpoint
elif CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
     print(f"Attempting to load model from checkpoint: {CHECKPOINT_PATH}")
     try:
         # We need the NeuroReportModel class definition available here
         eval_model = NeuroReportModel.load_from_checkpoint(CHECKPOINT_PATH, map_location='cpu') # Load to CPU first
         eval_tokenizer = eval_model.tokenizer # Tokenizer should be part of saved hparams or loaded manually
         model_to_eval_is_encoder_decoder = eval_model.is_encoder_decoder_model
         if eval_tokenizer is None:
              print("Tokenizer not found in checkpoint, attempting to load manually...")
              # Need model name from hparams or config
              model_name = eval_model.hparams.get('language_model_name', 'google/flan-t5-base') # Example fallback
              eval_tokenizer = AutoTokenizer.from_pretrained(model_name)
              if eval_tokenizer.pad_token is None: eval_tokenizer.pad_token = eval_tokenizer.eos_token
              eval_model.tokenizer = eval_tokenizer # Attach tokenizer if loaded manually
         print("Model loaded successfully from checkpoint.")
     except FileNotFoundError:
         print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
         eval_model = None
     except AttributeError:
          print("Error: Could not find NeuroReportModel class. Ensure stage6.py is accessible.")
          eval_model = None
     except Exception as e:
          print(f"Failed to load model from checkpoint: {e}")
          import traceback
          traceback.print_exc()
          eval_model = None
else:
    print("Error: No trained model instance found and no valid CHECKPOINT_PATH provided.")

# Setup DataLoader
# Use val_loader from training or a dedicated test_loader
if 'val_loader' in locals() and val_loader is not None:
    print("Using 'val_loader' for evaluation.")
    eval_loader = val_loader
# elif 'test_loader' in locals(): # Check for a dedicated test loader
#     eval_loader = test_loader
else:
    print("Warning: Creating dummy DataLoader for evaluation structure example.")
    # Create dummy dataset/loader if needed
    dummy_files = [f'dummy_test_{i}' for i in range(6)]
    dummy_labs = [{'question': f'Test Q{i}?', 'answer': f'Test Answer {i}.'} for i in range(6)] # VQA dummy
    # dummy_labs = [f'Test Report {i}' for i in range(6)] # Report dummy
    try:
        dummy_eval_dataset = MRIDataset(dummy_files, dummy_labs)
        eval_loader = DataLoader(dummy_eval_dataset, batch_size=EVAL_BATCH_SIZE)
        print(f"Created dummy evaluation DataLoader with {len(dummy_eval_dataset)} samples.")
    except Exception as e_load:
        print(f"Failed to create dummy eval loader: {e_load}")
        eval_loader = None


# --- Run Evaluation ---
if eval_model and eval_loader and eval_tokenizer:
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        print(f"Running evaluation on device: {eval_device}")
        eval_results, predictions, references = evaluate_pipeline(
            model=eval_model,
            dataloader=eval_loader,
            tokenizer=eval_tokenizer,
            device=eval_device,
            is_encoder_decoder=model_to_eval_is_encoder_decoder, # Use flag from loaded model
            max_gen_length=MAX_GEN_LENGTH_EVAL,
            num_beams=NUM_BEAMS_EVAL
        )

        print("\n--- Evaluation Results ---")
        if eval_results:
            for metric, value in eval_results.items():
                 # Format based on type (some rouge scores are tuples/objects)
                 if isinstance(value, float):
                     print(f"  {metric}: {value:.4f}")
                 else:
                     print(f"  {metric}: {value}") # Print raw value for complex scores
        else:
            print("  No metrics calculated.")


        # Print some examples
        print("\n--- Sample Predictions vs References ---")
        num_samples_to_show = 5
        for i in range(min(num_samples_to_show, len(predictions))):
            print(f"Sample {i+1}:")
            # Handle potential list wrapping if bleu_refs format was used
            ref_text = references[i][0] if isinstance(references[i], list) else references[i]
            print(f"  Reference: {ref_text}")
            print(f"  Prediction: {predictions[i]}\n")

    except Exception as e:
        import traceback
        print(f"Error during evaluation run: {e}")
        traceback.print_exc()
else:
    print("\nSkipping evaluation run because the model, dataloader, or tokenizer is not available.")
    if not eval_model: print("Reason: Model not loaded.")
    if not eval_loader: print("Reason: DataLoader not available.")
    if not eval_tokenizer: print("Reason: Tokenizer not available.")


print("\nStage 7: Evaluation setup complete.\n")

Add this in Stage 7: Evaluation and Validation

Goal: Evaluate the trained model's performance on a held-out test set using standard metrics.
Inputs: Trained NeuroReportModel, evaluation DataLoader, tokenizer.
Outputs: Dictionary containing calculated metrics (e.g., Accuracy, F1, BLEU, ROUGE, METEOR).