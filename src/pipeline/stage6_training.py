# ========== Stage 6: Training and Fine-Tuning ==========
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
    <file>src/pipeline