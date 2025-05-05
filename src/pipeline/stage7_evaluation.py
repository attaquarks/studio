# ========== Stage 7: Evaluation and Validation ==========
import torch
from torch.utils.data import DataLoader
from evaluate import load as load_metric # HuggingFace Evaluate library
from sklearn.metrics import accuracy_score # Corrected import
from rouge_score import rouge_scorer as rouge_scorer_lib # Import rouge scorer directly
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # For BLEU
import warnings
import numpy as np
import os
import json
import pytorch_lightning as pl # For loading checkpoint
from transformers import AutoTokenizer # For loading if needed
from typing import List, Dict, Tuple, Optional, Union

# --- Import Components from Previous Stages ---
# Need access to the main model class definition and dataloader definition
try:
    # Import the main NeuroReportModel class from Stage 6
    from .stage6_training import NeuroReportModel, MODEL_SAVE_PATH, MAX_LABEL_LENGTH
    # Import DataModule for creating eval dataloader if needed
    from .stage1_data_acquisition import NeuroReportDataModule, MRIDataset, BATCH_SIZE, TARGET_SIZE, NUM_SLICES_PER_SCAN
except ImportError:
    warnings.warn("Could not import NeuroReportModel/DataModule/constants from previous stages. Evaluation requires these definitions or a loaded checkpoint.")
    # Define placeholders if loading from checkpoint is not the primary goal
    MODEL_SAVE_PATH = "./neuroreport_model_checkpoint_dummy"
    MAX_LABEL_LENGTH = 256 # MUST match value used for training/generation
    BATCH_SIZE = 4
    TARGET_SIZE = (224, 224)
    NUM_SLICES_PER_SCAN = 64
    # If NeuroReportModel cannot be imported, loading from checkpoint won't work easily.
    class NeuroReportModel(pl.LightningModule): pass # Minimal placeholder
    class NeuroReportDataModule(pl.LightningDataModule): pass # Minimal placeholder
    class MRIDataset(torch.utils.data.Dataset): pass # Minimal placeholder


# --- Configuration for Stage 7 ---
# Path to the trained model checkpoint (if loading from file)
CHECKPOINT_PATH = None # Set to a specific .ckpt path, or None to use model instance from training run
# Example: Try to find the best checkpoint automatically if path not provided
if CHECKPOINT_PATH is None:
     try:
         # Try importing MODEL_SAVE_PATH from stage 6 if available
         from .stage6_training import MODEL_SAVE_PATH as CKPT_DIR_S6
         model_save_dir = CKPT_DIR_S6
     except ImportError:
         model_save_dir = MODEL_SAVE_PATH # Use placeholder path

     if os.path.isdir(model_save_dir):
          checkpoints = [f for f in os.listdir(model_save_dir) if f.endswith('.ckpt') and 'last' not in f.lower()]
          if checkpoints:
               # Sort checkpoints (heuristic: by name, assumes lower val_loss is better/earlier in alphabet)
               checkpoints.sort()
               CHECKPOINT_PATH = os.path.join(model_save_dir, checkpoints[0])
               print(f"Auto-selected checkpoint: {CHECKPOINT_PATH}")
          else: # Fallback to last.ckpt if no other checkpoints found
              last_ckpt_path = os.path.join(model_save_dir, 'last.ckpt')
              if os.path.exists(last_ckpt_path):
                   CHECKPOINT_PATH = last_ckpt_path
                   print(f"Using last checkpoint: {last_ckpt_path}")
              else:
                   print(f"No checkpoints found in {model_save_dir}. Cannot load model.")
     else:
          print(f"Checkpoint directory '{model_save_dir}' not found.")


EVAL_BATCH_SIZE = BATCH_SIZE * 2 # Use larger batch size for inference if memory allows
MAX_GEN_LENGTH_EVAL = MAX_LABEL_LENGTH # Max length for generation during evaluation
NUM_BEAMS_EVAL = 4 # Beam search for evaluation generation

# --- Load Metrics ---
# Use separate try-except blocks for robustness
print("Loading evaluation metrics...")
metrics_loaded = {}
try:
    # BLEU needs sacrebleu: pip install sacrebleu
    metrics_loaded['bleu'] = load_metric("bleu")
    print("Loaded BLEU.")
except Exception as e: print(f"Warning: Failed to load BLEU ('sacrebleu' might be needed): {e}")
try:
    # ROUGE needs rouge-score: pip install rouge-score
    metrics_loaded['rouge'] = load_metric("rouge")
    print("Loaded ROUGE.")
except Exception as e: print(f"Warning: Failed to load ROUGE ('rouge_score' might be needed): {e}")
try:
    # METEOR needs nltk: pip install nltk, then run nltk.download('wordnet') etc.
    metrics_loaded['meteor'] = load_metric("meteor")
    print("Loaded METEOR.")
except Exception as e: print(f"Warning: Failed to load METEOR ('nltk' data/omw-1.4 might be needed): {e}")

# --- Evaluator Class ---
class NeuroReportEvaluator:
    """Calculates evaluation metrics for NeuroReport outputs."""

    def __init__(self, mode="vqa"):
        """
        Initialize the evaluator.
        Args:
            mode: Evaluation mode ('vqa' or 'report')
        """
        self.mode = mode
        # Use the HuggingFace evaluate loaded metrics if available
        self.hf_rouge = metrics_loaded.get('rouge')
        self.hf_bleu = metrics_loaded.get('bleu')
        self.hf_meteor = metrics_loaded.get('meteor')
        # Keep direct ROUGE scorer as a fallback or for specific needs
        self.rouge_scorer = rouge_scorer_lib.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def evaluate_vqa(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluate VQA outputs."""
        results = {}
        if not predictions or not references or len(predictions) != len(references):
            warnings.warn("Invalid input for VQA evaluation.")
            return results

        # 1. Exact Match Accuracy
        norm_preds = [p.strip().lower() for p in predictions]
        norm_refs = [r.strip().lower() for r in references]
        try:
             results['vqa_accuracy_exact_match'] = accuracy_score(norm_refs, norm_preds)
        except Exception as e: print(f"Error calculating accuracy: {e}")


        # 2. BLEU Score (Treating answers as short sentences)
        if self.hf_bleu:
            try:
                bleu_preds_vqa = [p.split() for p in predictions]
                bleu_refs_vqa = [[r.split()] for r in references] # List of lists format
                bleu_score = self.hf_bleu.compute(predictions=bleu_preds_vqa, references=bleu_refs_vqa)
                results['vqa_bleu'] = bleu_score.get('bleu', 0.0) if bleu_score else 0.0
            except Exception as e: print(f"Error calculating HF BLEU for VQA: {e}")
        else: # Fallback NLTK BLEU
             try:
                  bleu_scores_nltk = []
                  for pred, ref in zip(predictions, references):
                       p_tok, r_tok = pred.split(), [ref.split()]
                       score = sentence_bleu(r_tok, p_tok, smoothing_function=self.smoothing) if p_tok else 0.0
                       bleu_scores_nltk.append(score)
                  results['vqa_bleu_nltk'] = np.mean(bleu_scores_nltk) if bleu_scores_nltk else 0.0
             except Exception as e: print(f"Error calculating NLTK BLEU for VQA: {e}")


        # 3. ROUGE Score
        if self.hf_rouge:
             try:
                 rouge_score = self.hf_rouge.compute(predictions=predictions, references=references)
                 results.update({f'vqa_{k}': v for k,v in rouge_score.items()}) # Add vqa_rouge1 etc.
             except Exception as e: print(f"Error calculating HF ROUGE for VQA: {e}")
        else: # Fallback direct ROUGE scorer
            try:
                rouge_agg = {'rouge1': [], 'rouge2': [], 'rougeL': []}
                for pred, ref in zip(predictions, references):
                    scores = self.rouge_scorer.score(ref, pred)
                    rouge_agg['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_agg['rouge2'].append(scores['rouge2'].fmeasure)
                    rouge_agg['rougeL'].append(scores['rougeL'].fmeasure)
                results['vqa_rouge1_manual'] = np.mean(rouge_agg['rouge1']) if rouge_agg['rouge1'] else 0.0
                results['vqa_rouge2_manual'] = np.mean(rouge_agg['rouge2']) if rouge_agg['rouge2'] else 0.0
                results['vqa_rougeL_manual'] = np.mean(rouge_agg['rougeL']) if rouge_agg['rougeL'] else 0.0
            except Exception as e: print(f"Error calculating manual ROUGE for VQA: {e}")

        return results

    def evaluate_report(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluate report generation outputs."""
        results = {}
        if not predictions or not references or len(predictions) != len(references):
            warnings.warn("Invalid input for Report evaluation.")
            return results

        # BLEU
        if self.hf_bleu:
            try:
                bleu_preds_rep = [p.split() for p in predictions]
                bleu_refs_rep = [[r.split()] for r in references] # List of lists format
                bleu_score = self.hf_bleu.compute(predictions=bleu_preds_rep, references=bleu_refs_rep)
                results['report_bleu'] = bleu_score.get('bleu', 0.0) if bleu_score else 0.0
            except Exception as e: print(f"Error calculating HF BLEU for Report: {e}")

        # ROUGE
        if self.hf_rouge:
            try:
                rouge_score = self.hf_rouge.compute(predictions=predictions, references=references)
                results.update({f'report_{k}': v for k,v in rouge_score.items()})
            except Exception as e: print(f"Error calculating HF ROUGE for Report: {e}")

        # METEOR
        if self.hf_meteor:
            try:
                # Ensure NLTK data is available
                try:
                    import nltk
                    nltk.data.find('corpora/wordnet')
                    nltk.data.find('corpora/omw-1.4')
                except LookupError:
                    print("Downloading NLTK data for METEOR (wordnet, omw-1.4)...")
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)

                meteor_score = self.hf_meteor.compute(predictions=predictions, references=references)
                results['report_meteor'] = meteor_score.get('meteor', 0.0) if meteor_score else 0.0
            except Exception as e: print(f"Error calculating HF METEOR for Report: {e}")

        # TODO: Add CheXbert or other clinical accuracy metrics if needed

        return results


    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluate based on the mode."""
        print(f"Calculating metrics for mode: {self.mode}")
        if self.mode == "vqa":
            return self.evaluate_vqa(predictions, references)
        elif self.mode == "report":
            return self.evaluate_report(predictions, references)
        else:
            warnings.warn(f"Unknown evaluation mode: {self.mode}")
            return {}


# --- Evaluation Function (Main Loop) ---
@torch.no_grad() # Disable gradients for inference
def evaluate_pipeline(
    model: NeuroReportModel,
    dataloader: DataLoader,
    device: torch.device,
    max_gen_length: int = MAX_GEN_LENGTH_EVAL,
    num_beams: int = NUM_BEAMS_EVAL
) -> Tuple[Dict, List[str], List[str]]:
    """
    Evaluates the full NeuroReport model on a given dataloader.

    Args:
        model (NeuroReportModel): The trained/loaded NeuroReportModel instance.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to run evaluation on ('cuda' or 'cpu').
        max_gen_length (int): Max *new* tokens for generated output.
        num_beams (int): Number of beams for generation.

    Returns:
        tuple: (dict: calculated metrics, list: all predictions, list: all references)
    """
    if not all(hasattr(model, attr) for attr in ['vision_encoder', 'slice_aggregator', 'cross_attention_bridge', 'language_decoder']):
        raise AttributeError("Model instance is missing required components.")
    if model.language_decoder.model is None or model.language_decoder.tokenizer is None:
         raise ValueError("Language model or tokenizer is missing within the model's language_decoder. Cannot evaluate.")

    model.eval() # Set model to evaluation mode
    model.to(device)
    tokenizer = model.language_decoder.tokenizer
    eval_mode = model.mode # Get mode ('vqa' or 'report') from the loaded model
    evaluator = NeuroReportEvaluator(mode=eval_mode) # Instantiate evaluator

    all_predictions = []
    all_references = []

    print(f"Starting evaluation (mode: {eval_mode}) on {len(dataloader.dataset)} samples using device {device}...")
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        print(f"\rProcessing batch {batch_count}/{len(dataloader)}...", end="")
        pixel_values = batch['pixel_values'].to(device)
        input_ids, attention_mask = None, None # Initialize

        # Prepare references and inputs
        if eval_mode == "vqa":
            if 'question' not in batch or 'answer' not in batch: continue
            references = batch['answer']
            input_texts = [f"question: {q} context: " for q in batch['question']] # VQA format
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        elif eval_mode == "report":
            if 'report' not in batch: continue
            references = batch['report']
            if model.language_decoder.model_type == 'seq2seq':
                input_texts = ["generate report: "] * len(references)
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            else: # CausalLM
                # Start with BOS token
                bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1 # Handle missing BOS
                if bos_token_id == -1: warnings.warn("Tokenizer missing BOS token for CausalLM generation start."); continue
                input_ids = torch.full((pixel_values.shape[0], 1), bos_token_id, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
        else: continue # Should not happen if mode is validated

        all_references.extend(references) # Store ground truth

        # --- Inference ---
        try:
            visual_features_slices = model.vision_encoder(pixel_values)
            aggregated_visual_features = model.slice_aggregator(visual_features_slices)

            # Prepare generation arguments for language_decoder.generate
            gen_args = {
                "max_new_tokens": max_gen_length,
                "num_beams": num_beams,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }

            # Handle bridging and prepare inputs for generate based on bridge type
            if hasattr(model, 'cross_attention_bridge') and isinstance(model.cross_attention_bridge, nn.Module) and not isinstance(model.cross_attention_bridge, nn.Identity):
                 # --- Using CrossAttentionBridge ---
                 # Get language embeddings first
                 if model.language_decoder.model_type == 'seq2seq':
                     if hasattr(model.language_decoder.model, 'get_input_embeddings'):
                          lang_embeds = model.language_decoder.model.get_input_embeddings()(input_ids)
                     else: lang_embeds = model.language_decoder.model.shared(input_ids)
                 else: # CausalLM
                      lang_embeds = model.language_decoder.model.get_input_embeddings()(input_ids)

                 # Apply bridge
                 enhanced_features = model.cross_attention_bridge(
                      aggregated_visual_features, lang_embeds,
                      language_attention_mask=attention_mask.bool() if attention_mask is not None else None
                 )
                 gen_args["inputs_embeds"] = enhanced_features
                 gen_args["attention_mask"] = attention_mask # Still needed for position info
            else:
                 # --- Using Simple Projection or No Bridge ---
                 gen_args["input_ids"] = input_ids
                 gen_args["attention_mask"] = attention_mask
                 if model.language_decoder.model_type == 'seq2seq':
                      # Apply projection if bridge exists and is not Identity
                      if hasattr(model, 'cross_attention_bridge') and isinstance(model.cross_attention_bridge, nn.Linear):
                           conditioned_visual = model.cross_attention_bridge(aggregated_visual_features)
                      else: # Assume no projection needed or handled elsewhere
                           conditioned_visual = aggregated_visual_features
                      # Pass as encoder_outputs
                      gen_args["encoder_outputs"] = (conditioned_visual.unsqueeze(1),) # Shape [B, 1, D_agg/D_l]


            # Generate
            output_sequences = model.language_decoder.generate(**gen_args)

            # Decode predictions
            # CausalLMs might include prompt in output, remove it
            prompt_len_to_remove = 0
            if model.language_decoder.model_type == 'causal_lm' and input_ids.shape[1] > 0:
                 # Check if output starts with input (common case)
                 if output_sequences.shape[1] >= input_ids.shape[1] and torch.equal(output_sequences[:, :input_ids.shape[1]], input_ids):
                      prompt_len_to_remove = input_ids.shape[1]

            predictions = model.language_decoder.decode(output_sequences[:, prompt_len_to_remove:])
            all_predictions.extend(predictions)

        except Exception as e_infer:
             print(f"\nError during inference batch {batch_count}: {e_infer}")
             all_predictions.extend(["INFERENCE_ERROR"] * len(references))


    print("\nGeneration complete. Calculating metrics...")
    # Filter out errors before metric calculation
    valid_indices = [i for i, p in enumerate(all_predictions) if p != "INFERENCE_ERROR"]
    filtered_predictions = [all_predictions[i] for i in valid_indices]
    filtered_references = [all_references[i] for i in valid_indices]

    results = {}
    if filtered_predictions:
        results = evaluator.evaluate(filtered_predictions, filtered_references)
    else:
        print("Warning: No valid predictions available to calculate metrics.")

    print("\nEvaluation Metrics Calculation Complete.")
    # Return metrics dict, raw predictions, and raw references
    return results, all_predictions, all_references


# --- Example Evaluation Usage ---
if __name__ == "__main__":
    print("--- Stage 7 Example ---")

    eval_model_instance = None
    eval_dataloader_instance = None

    # Try to load model
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
         print(f"Attempting to load model from checkpoint: {CHECKPOINT_PATH}")
         try:
             eval_model_instance = NeuroReportModel.load_from_checkpoint(CHECKPOINT_PATH, map_location='cpu')
             print("Model loaded successfully.")
         except Exception as e: print(f"Failed to load model: {e}")
    # elif 'neuro_report_model_instance' in locals() and isinstance(neuro_report_model_instance, NeuroReportModel):
    #      eval_model_instance = neuro_report_model_instance # Use if available from previous run

    # Setup DataLoader (using dummy data if necessary)
    DUMMY_DATA_DIR_S7 = "./dummy_mri_data_s7"
    DUMMY_ANNOTATIONS_FILE_S7 = "./dummy_annotations_s7.csv"
    if not os.path.exists(DUMMY_ANNOTATIONS_FILE_S7):
        # Create dummy data...
         os.makedirs(DUMMY_DATA_DIR_S7, exist_ok=True)
         import pandas as pd
         import nibabel as nib
         dummy_data_s7 = [{'file_name': f'scan_{i}.nii.gz', 'question': f'Q{i}', 'answer': f'A{i}', 'report': f'R{i}'} for i in range(6)]
         for item in dummy_data_s7:
             fp = os.path.join(DUMMY_DATA_DIR_S7, item['file_name'])
             nib.save(nib.Nifti1Image(np.random.rand(16, 16, 5).astype(np.float32), np.eye(4)), fp)
         pd.DataFrame(dummy_data_s7).to_csv(DUMMY_ANNOTATIONS_FILE_S7, index=False)
         print("Created dummy data for Stage 7")


    try:
        eval_data_module = NeuroReportDataModule(
            data_dir=DUMMY_DATA_DIR_S7, annotations_path=DUMMY_ANNOTATIONS_FILE_S7,
            batch_size=EVAL_BATCH_SIZE, mode=eval_model_instance.mode if eval_model_instance else "vqa",
            target_size=TARGET_SIZE, val_split=0.0, test_split=1.0, num_workers=0, n_slices=NUM_SLICES_PER_SCAN
        )
        eval_data_module.prepare_data()
        eval_data_module.setup('test')
        eval_dataloader_instance = eval_data_module.test_dataloader()
    except Exception as e_dm: print(f"Failed to create DataLoader: {e_dm}")

    # --- Run Evaluation ---
    if eval_model_instance and eval_dataloader_instance:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            print(f"\nRunning evaluation on device: {eval_device}...")
            eval_results, predictions, references = evaluate_pipeline(
                model=eval_model_instance,
                dataloader=eval_dataloader_instance,
                device=eval_device,
                max_gen_length=MAX_GEN_LENGTH_EVAL,
                num_beams=NUM_BEAMS_EVAL
            )

            print("\n--- Evaluation Results ---")
            print(json.dumps(eval_results, indent=2))

            print("\n--- Sample Predictions vs References ---")
            num_samples_to_show = min(5, len(predictions))
            if num_samples_to_show > 0:
                 for i in range(num_samples_to_show): print(f"Ref {i+1}: {references[i]}\nPred{i+1}: {predictions[i]}\n---")
            else: print("No predictions/references to display.")

        except Exception as e: import traceback; print(f"\nError during evaluation run: {e}"); traceback.print_exc()
    else:
        print("\nSkipping evaluation run (model or dataloader not available).")

    # Clean up dummy files...

    print("\nStage 7: Evaluation setup complete.\n")
