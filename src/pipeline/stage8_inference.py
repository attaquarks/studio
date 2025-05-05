# ========== Stage 8: Inference - Interactive Demo ==========
import torch
import pytorch_lightning as pl
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import warnings
from typing import Optional, Tuple

# --- Import Components ---
# Need the main model class and preprocessor definition
try:
    from .stage6_training import NeuroReportModel # The main LightningModule
    from .stage1_data_acquisition import MRIPreprocessor, TARGET_SIZE, NUM_SLICES_PER_SCAN # Preprocessor and config
    from .stage4_vision_language_bridge import CrossAttentionBridge
except ImportError:
    warnings.warn("Could not import NeuroReportModel/MRIPreprocessor/constants. Inference demo requires these.")
    # Define minimal placeholders if necessary for script structure
    class NeuroReportModel(pl.LightningModule): pass
    class MRIPreprocessor:
        def __init__(self, target_size=(224, 224), **kwargs): self.target_size = target_size
        def process_volume(self, file_path): return np.random.rand(64, target_size[0], target_size[1]).astype(np.float32)
    TARGET_SIZE = (224, 224)
    NUM_SLICES_PER_SCAN = 64
    class CrossAttentionBridge: pass

# --- Configuration ---
DEFAULT_MODEL_PATH = None # Set to the path of your best *.ckpt file
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SLICES_TO_VISUALIZE = 9 # How many slices to show in the demo output

# --- Inference Class ---
class NeuroReportInference:
    """Handles loading the model and running inference for the demo."""

    def __init__(self, model_path: Optional[str] = DEFAULT_MODEL_PATH, device: str = DEFAULT_DEVICE):
        self.device = torch.device(device)
        self.model_path = model_path

        if self.model_path is None:
             # Auto-find checkpoint logic (same as Stage 7)
             try:
                 from .stage6_training import MODEL_SAVE_PATH
                 if os.path.isdir(MODEL_SAVE_PATH):
                     # Find best checkpoint... (implement logic as in Stage 7)
                     checkpoints = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.ckpt') and 'last' not in f.lower()]
                     if checkpoints:
                         checkpoints.sort()
                         self.model_path = os.path.join(MODEL_SAVE_PATH, checkpoints[0])
                     else: # Fallback
                         last_ckpt = os.path.join(MODEL_SAVE_PATH, 'last.ckpt')
                         if os.path.exists(last_ckpt): self.model_path = last_ckpt
             except ImportError: pass # Ignore if MODEL_SAVE_PATH not found
             except Exception as e: print(f"Error auto-finding checkpoint: {e}")


        if not self.model_path or not os.path.exists(self.model_path):
             raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        try:
            # Load model to CPU first, then move
            self.model = NeuroReportModel.load_from_checkpoint(self.model_path, map_location='cpu')
            self.model.eval()
            self.model.to(self.device)
            print(f"Model loaded to {self.device}.")
            self.mode = self.model.mode
            print(f"Model mode: {self.mode}")
            # Extract necessary hparams for preprocessing and inference
            self.target_size = getattr(self.model.hparams, 'target_size', TARGET_SIZE)
            self.normalization = getattr(self.model.hparams, 'normalization', "zero_mean_unit_var")
            self.n_slices = getattr(self.model.hparams, 'n_slices', NUM_SLICES_PER_SCAN)
            self.max_label_length = getattr(self.model.hparams, 'max_label_length', 256) # Max length for generation
            self.tokenizer = self.model.language_decoder.tokenizer # Get tokenizer
        except Exception as e:
            print(f"Error loading model: {e}"); raise

        self.preprocessor = MRIPreprocessor(target_size=self.target_size, normalization=self.normalization)
        print(f"Preprocessor: target_size={self.target_size}, normalization='{self.normalization}'")

    def _select_pad_inference_slices(self, slices: np.ndarray) -> np.ndarray:
         """Selects/pads slices for inference."""
         target_slices = self.n_slices # Use n_slices from loaded model hparams
         current_slices = slices.shape[0]
         if current_slices == 0: return np.zeros((target_slices, *self.target_size), dtype=np.float32)
         if current_slices == target_slices: return slices
         if current_slices > target_slices:
              center = current_slices // 2
              start = max(0, center - target_slices // 2)
              end = start + target_slices
              return slices[start:end, :, :]
         else: # Pad
              pad_before = (target_slices - current_slices) // 2
              pad_after = target_slices - current_slices - pad_before
              return np.pad(slices, [(pad_before, pad_after), (0, 0), (0, 0)], mode='constant')

    def preprocess_and_batch(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """Preprocesses NIfTI file, pads/selects slices, adds batch/channel dims."""
        try:
            slices_np = self.preprocessor.process_volume(file_path) # (N_orig, H, W)
            if slices_np.size == 0: return None, None
            slices_np = self._select_pad_inference_slices(slices_np) # (n_slices, H, W)
            slices_tensor = torch.from_numpy(slices_np).float().unsqueeze(0).unsqueeze(2) # (1, n_slices, 1, H, W)
            if slices_tensor.shape[2] == 1: slices_tensor = slices_tensor.repeat(1, 1, 3, 1, 1) # (1, n_slices, 3, H, W)
            return slices_tensor.to(self.device), slices_np
        except Exception as e: print(f"Error preprocessing {file_path}: {e}"); return None, None

    def visualize_slices(self, slices_np: np.ndarray, num_slices: int = NUM_SLICES_TO_VISUALIZE) -> Optional[Image.Image]:
        """Visualizes MRI slices."""
        if slices_np is None or slices_np.size == 0: return None
        num_available = slices_np.shape[0]
        indices = np.linspace(0, num_available - 1, min(num_slices, num_available)).astype(int)
        selected_slices = slices_np[indices]
        grid_size = int(np.ceil(np.sqrt(len(selected_slices))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6)) # Smaller figure
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < len(selected_slices):
                slice_to_show = selected_slices[i]
                # Simple windowing for visualization (adjust percentile values if needed)
                p_low, p_high = np.percentile(slice_to_show, [1, 99])
                slice_to_show = np.clip(slice_to_show, p_low, p_high)
                if p_high > p_low: slice_to_show = (slice_to_show - p_low) / (p_high - p_low)
                ax.imshow(slice_to_show, cmap='gray')
                ax.set_title(f"Sl {indices[i]}", fontsize=8) # Smaller title
                ax.axis('off')
            else: ax.axis('off')
        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        return img

    @torch.no_grad()
    def predict(self, file_obj, question: Optional[str] = None) -> Tuple[Optional[Image.Image], str]:
         """Generic prediction function for Gradio."""
         if file_obj is None: return None, "Please upload an MRI file."

         file_path = file_obj.name
         is_vqa_request = bool(question) # Check if a question was provided
         expected_mode = "vqa" if is_vqa_request else "report"

         if self.mode != expected_mode:
              return None, f"Model loaded in '{self.mode}' mode, but request is for '{expected_mode}'."

         print(f"Processing {expected_mode} for: {file_path}")
         slices_tensor, slices_np = self.preprocess_and_batch(file_path)
         if slices_tensor is None: return None, "Error processing the MRI file."

         viz_img = self.visualize_slices(slices_np)

         # Prepare inputs for generation
         input_ids, attention_mask = None, None
         if is_vqa_request:
              input_texts = [f"question: {question} context: "] # VQA format
              inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
              input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
         else: # Report generation
              if self.model.language_decoder.model_type == 'seq2seq':
                   input_texts = ["generate report: "]
                   inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                   input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
              else: # CausalLM
                   input_ids = torch.full((1, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
                   attention_mask = torch.ones_like(input_ids)

         # Perform inference using model components
         try:
            visual_features_slices = self.model.vision_encoder(slices_tensor)
            aggregated_visual_features = self.model.slice_aggregator(visual_features_slices)

            # Prepare args for language_decoder.generate
            gen_args = {
                "max_new_tokens": 512 if not is_vqa_request else 128, # Longer for reports
                "num_beams": NUM_BEAMS_EVAL,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }

            # Handle bridging based on bridge type
            if isinstance(self.model.cross_attention_bridge, CrossAttentionBridge):
                 # Get language embeddings
                 if self.model.language_decoder.model_type == 'seq2seq':
                      if hasattr(self.model.language_decoder.model, 'get_input_embeddings'):
                           lang_embeds = self.model.language_decoder.model.get_input_embeddings()(input_ids)
                      else: lang_embeds = self.model.language_decoder.model.shared(input_ids)
                 else: # CausalLM
                      lang_embeds = self.model.language_decoder.model.get_input_embeddings()(input_ids)

                 # Apply bridge
                 enhanced_features = self.model.cross_attention_bridge(
                      aggregated_visual_features, lang_embeds,
                      language_attention_mask=attention_mask.bool() if attention_mask is not None else None
                 )
                 gen_args["inputs_embeds"] = enhanced_features
                 gen_args["attention_mask"] = attention_mask # Still needed
            else: # Identity or simple projection bridge
                 gen_args["input_ids"] = input_ids
                 gen_args["attention_mask"] = attention_mask
                 if self.model.language_decoder.model_type == 'seq2seq':
                      conditioned_visual = self.model.cross_attention_bridge(aggregated_visual_features)
                      gen_args["encoder_outputs"] = (conditioned_visual.unsqueeze(1),)

            # Generate
            generated_ids = self.model.language_decoder.generate(**gen_args)
            result_text = self.model.language_decoder.decode(generated_ids)[0]

         except Exception as e:
             print(f"Error during generation: {e}"); import traceback; traceback.print_exc()
             result_text = "Error generating output."

         return viz_img, result_text


def launch_demo(model_path: str, device: str = DEFAULT_DEVICE):
    """Launches the Gradio interface."""
    try:
        inference_handler = NeuroReportInference(model_path=model_path, device=device)
    except Exception as e: print(f"Failed to initialize inference handler: {e}"); return

    # --- Define Gradio Blocks ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NeuroReport: Medical Imaging Analysis")
        gr.Markdown("Upload a NIfTI MRI scan (.nii or .nii.gz) and either ask a question (VQA) or generate a report.")

        with gr.Row():
            with gr.Column(scale=1):
                mri_file_input = gr.File(label="Upload MRI Scan (NIfTI)", type="filepath")
                question_input = gr.Textbox(label="Question (Optional)", placeholder="Ask a question for VQA mode...")
                submit_button = gr.Button("Analyze Scan", variant="primary")
                # Hidden input to determine mode based on question presence
                mode_indicator = gr.Textbox(value=inference_handler.mode, label="Detected Model Mode", interactive=False)

            with gr.Column(scale=2):
                output_slices_viz = gr.Image(label="MRI Slices Visualization", type="pil")
                output_text = gr.Textbox(label="Result (Answer or Report)", lines=15)

        # --- Logic for Submission ---
        def handle_submit(file_obj, question_text):
            # Call the unified predict function
            return inference_handler.predict(file_obj, question=question_text if question_text else None)

        submit_button.click(
            fn=handle_submit,
            inputs=[mri_file_input, question_input],
            outputs=[output_slices_viz, output_text]
        )

        # --- Examples ---
        gr.Examples(
            examples=[
                ["./dummy_mri_data_s7/scan_1.nii.gz", "Is there evidence of a tumor?"], # VQA example
                ["./dummy_mri_data_s7/scan_1.nii.gz", None], # Report generation example
            ],
            inputs=[mri_file_input, question_input],
            outputs=[output_slices_viz, output_text],
            fn=handle_submit,
            cache_examples=False # Re-run examples on click
        )

    print("Launching Gradio demo...")
    demo.launch(share=False) # Set share=True for public link


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Stage 8: Inference Demo ---")

    # Determine checkpoint path (use logic from Stage 7/auto-find)
    ckpt_path_to_launch = None
    try:
         from .stage6_training import MODEL_SAVE_PATH
         model_save_dir = MODEL_SAVE_PATH
         if os.path.isdir(model_save_dir):
              checkpoints = [f for f in os.listdir(model_save_dir) if f.endswith('.ckpt') and 'last' not in f.lower()]
              if checkpoints: checkpoints.sort(); ckpt_path_to_launch = os.path.join(model_save_dir, checkpoints[0])
              else:
                   last_ckpt = os.path.join(model_save_dir, 'last.ckpt')
                   if os.path.exists(last_ckpt): ckpt_path_to_launch = last_ckpt
         else: print(f"Checkpoint directory not found: {model_save_dir}")
    except ImportError: print("MODEL_SAVE_PATH not imported, cannot auto-find checkpoint.")
    except Exception as e: print(f"Error finding checkpoint: {e}")

    # Override with specific path if needed
    # ckpt_path_to_launch = "/path/to/your/specific/checkpoint.ckpt"

    if ckpt_path_to_launch and os.path.exists(ckpt_path_to_launch):
         print(f"Launching demo with checkpoint: {ckpt_path_to_launch}")
         # Create dummy file for examples if needed
         if not os.path.exists("./dummy_mri_data_s7/scan_1.nii.gz"):
              os.makedirs("./dummy_mri_data_s7", exist_ok=True)
              nib.save(nib.Nifti1Image(np.random.rand(16,16,5).astype(np.float32), np.eye(4)), "./dummy_mri_data_s7/scan_1.nii.gz")
         launch_demo(model_path=ckpt_path_to_launch)
    else:
         print("\nNo valid checkpoint path found. Cannot launch demo.")
         print(f"Checked path: {ckpt_path_to_launch}")
         print("Please train a model (Stage 9) or provide a valid checkpoint path.")

    print("\nStage 8: Inference Demo setup complete.\n")
