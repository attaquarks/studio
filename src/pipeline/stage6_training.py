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
    # Import DataModule definition, including split-related configs
    from .stage1_data_acquisition import NeuroReportDataModule, BATCH_SIZE, TARGET_SIZE, NUM_SLICES_PER_SCAN, VAL_SPLIT, TEST_SPLIT
except ImportError:
    warnings.warn("Could not import from stage1. Defining dummy components for Stage 6 structure.")
    BATCH_SIZE = 2
    NUM_SLICES_PER_SCAN = 8
    TARGET_SIZE=(32, 32) # Small dummy size
    VAL_SPLIT, TEST_SPLIT = 0.1, 0.1 # Dummy splits
    class NeuroReportDataModule(pl.LightningDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.batch_size = BATCH_SIZE
            self._seed = 42
            self.val_split=VAL_SPLIT
            self.test_split=TEST_SPLIT

        def setup(self, stage=None):
            # Minimal setup with dummy data
            class DummyDataset(torch.utils.data.Dataset):
                def __init__(self, length=20): self.len=length
                def __len__(self): return self.len
                def __getitem__(self, idx):
                     item = {'pixel_values': torch.randn(NUM_SLICES_PER_SCAN, 3, TARGET_SIZE[0], TARGET_SIZE[1])}
                     if self.mode == 'vqa': item.update({'question': f'Q{idx}', 'answer': f'A{idx}'})
                     else: item['report'] = f'R{idx}'
                     return item

            self.full_dataset = DummyDataset(length=20) # Example full size
            indices = list(range(len(self.full_dataset)))
            num_test = int(self.test_split * len(indices))
            num_val = int(self.val_split * (len(indices) - num_test))
            train_indices, temp_indices = train_test_split(indices, train_size=len(indices)-num_val-num_test, random_state=self._seed)
            if num_val > 0 and num_test > 0: val_indices, test_indices = train_test_split(temp_indices, train_size=num_val, random_state=self._seed)
            elif num_val > 0: val_indices, test_indices = temp_indices, []
            else: val_indices, test_indices = [], temp_indices

            if stage == 'fit' or stage is None:
                self.train_dataset = Subset(self.full_dataset, train_indices)
                self.val_dataset = Subset(self.full_dataset, val_indices) if val_indices else None
            if stage == 'test' or stage is None:
                self.test_dataset = Subset(self.full_dataset, test_indices) if test_indices else None

        def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.batch_size) if hasattr(self, 'train_dataset') else None
        def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.batch_size) if hasattr(self, 'val_dataset') and self.val_dataset else None
        def test_dataloader(self): return DataLoader(self.test_dataset, batch_size=self.batch_size) if hasattr(self, 'test_dataset') and self.test_dataset else None


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
    is_encoder_decoder = STAGE5_LM_TYPE == 'seq2seq' # Use determined type
except ImportError as e:
    warnings.warn(f"Could not import components from stage5 ({e}). Defining dummy components for Stage 6 structure.")
    LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Use a consistent model name
    STAGE5_LM_TYPE = 'seq2seq'
    is_encoder_decoder = True # Consistent with T5
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
    # Attempt to reuse components if they exist from previous runs/cells
    if 'vision_encoder' not in locals(): vision_encoder = VisionEncoder()
    if 'slice_aggregator' not in locals(): slice_aggregator = SliceAggregator(feature_dim=vision_encoder.feature_dim)
    if 'language_decoder' not in locals(): language_decoder = LanguageDecoder()
    is_encoder_decoder = language_decoder.model_type == 'seq2seq' # Get type from instance
    tokenizer_instance = language_decoder.tokenizer # Get tokenizer from instance

    # Determine bridge necessity
    aggregator_output_dim = slice_aggregator.output_dim
    language_model_dim = language_decoder.get_model_dim()
    if 'bridge' not in locals():
         if aggregator_output_dim != language_model_dim:
              print(f"Dimensions mismatch: Aggregator ({aggregator_output_dim}) != LM ({language_model_dim}). Using VisionLanguageBridge.")
              bridge_instance = VisionLanguageBridge(visual_dim=aggregator_output_dim, language_dim=language_model_dim)
         else:
              print("Dimensions match. Using Identity bridge.")
              bridge_instance = nn.Identity()
    else: # Use existing bridge if available
        bridge_instance = bridge

except NameError as e:
    print(f"Error: Component missing. Ensure previous stages are run or dummy components are created. {e}") ; exit()
except Exception as e_inst:
    print(f"Error instantiating pipeline components for training: {e_inst}")
    print("Check component definitions and configurations in previous stages."); exit()


# --- Combined Model (PyTorch Lightning Module) ---
class NeuroReportModel(pl.LightningModule):
    """
    Combines all stages into a single model for end-to-end training using PyTorch Lightning.
    Includes validation step.
    """
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 slice_aggregator: SliceAggregator,
                 bridge: nn.Module, # Can be VisionLanguageBridge or nn.Identity
                 language_decoder: LanguageDecoder,
                 tokenizer: AutoTokenizer, # Pass tokenizer
                 is_encoder_decoder_model: bool, # Pass model type flag
                 mode: str = "report", # Default to report as VQA needs questions
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 warmup_steps_ratio: float = WARMUP_STEPS_RATIO,
                 max_label_length: int = MAX_LABEL_LENGTH,
                 **kwargs): # Absorb other potential hparams from config
        super().__init__()
        # Save hyperparameters - important for loading checkpoints
        # Ignore large components to avoid saving them directly in hparams.yaml
        self.save_hyperparameters(ignore=['vision_encoder', 'slice_aggregator', 'bridge', 'language_decoder', 'tokenizer'])

        self.vision_encoder = vision_encoder
        self.slice_aggregator = slice_aggregator
        self.bridge = bridge # Can be VisionLanguageBridge or nn.Identity
        self.language_decoder = language_decoder # Contains model
        self.tokenizer = tokenizer # Store tokenizer
        self.is_encoder_decoder_model = is_encoder_decoder_model
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
        slice_features = self.vision_encoder(pixel_values) # (B, N, VisFeatDim)

        # Stage 3: Slice Aggregation
        scan_embedding = self.slice_aggregator(slice_features) # (B, AggFeatDim)

        # Stage 4: Bridging (Projection)
        conditioned_embedding = self.bridge(scan_embedding) # (B, LangModelDim)

        # Stage 5: Language Model Processing
        model_inputs = {"return_dict": True}

        if self.is_encoder_decoder_model:
            # Reshape visual embedding for encoder_outputs: (B, 1, LangModelDim)
            encoder_hidden_states = conditioned_embedding.unsqueeze(1)
            model_inputs["encoder_outputs"] = (encoder_hidden_states,)
            if input_ids is not None: model_inputs["input_ids"] = input_ids # decoder_input_ids implicitly
            if attention_mask is not None: model_inputs["attention_mask"] = attention_mask
            if labels is not None: model_inputs["labels"] = labels

        else: # CausalLM
            # Basic CausalLM - needs input_ids (e.g., BOS token or prompt)
            if input_ids is None: raise ValueError("input_ids are required for CausalLM forward pass during training.")
            # Conditioning is tricky - this basic version might ignore visual embedding.
            # For true multimodal, need to construct `inputs_embeds` by combining
            # projected visual embedding and text token embeddings.
            warnings.warn("CausalLM forward pass uses text input only. Modify to include visual conditioning.")
            model_inputs["input_ids"] = input_ids
            if attention_mask is not None: model_inputs["attention_mask"] = attention_mask
            if labels is not None: model_inputs["labels"] = labels # Shifted automatically by HF

        # Check if we have labels to calculate loss
        if "labels" not in model_inputs or model_inputs["labels"] is None:
            # If no labels, run inference pass (e.g., just get logits)
            model_inputs.pop("labels", None)
            with torch.no_grad(): outputs = self.language_decoder.model(**model_inputs)
            outputs.loss = None
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

        # --- Determine inputs and targets based on mode and available keys ---
        if self.mode == "vqa":
            if 'question' in batch and 'answer' in batch:
                 input_texts = [f"question: {q} context: " for q in batch['question']]
                 target_texts = batch['answer']
            else: # Handle missing VQA annotations
                 warnings.warn(f"VQA mode selected, but 'question' or 'answer' missing in batch.")
                 return pixel_values, None, None, None # Return None for texts/labels

        elif self.mode == "report":
            if 'report' in batch: # Use report if available
                 target_texts = batch['report']
                 if self.is_encoder_decoder_model:
                      input_texts = ["generate report: "] * batch_size
                 else: # CausalLM - needs starting token(s)
                      input_texts = [self.tokenizer.bos_token] * batch_size if self.tokenizer.bos_token else [""] * batch_size
            else: # Handle missing report annotations
                 warnings.warn(f"Report mode selected, but 'report' missing in batch.")
                 return pixel_values, None, None, None # Return None for texts/labels
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vqa' or 'report'.")

        # --- Tokenize inputs if available ---
        if input_texts:
            try:
                input_encoding = self.tokenizer(
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
                target_encoding = self.tokenizer(
                    target_texts, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_label_length
                )
                labels = target_encoding.input_ids.to(self.device)
                # Replace padding token id in labels with -100 for CrossEntropyLoss
                labels[labels == self.tokenizer.pad_token_id] = -100
            except Exception as e_tok_tgt:
                 warnings.warn(f"Error tokenizing target texts: {e_tok_tgt}")
                 labels = None # Mark as failed

        return pixel_values, input_ids, attention_mask, labels

    def training_step(self, batch, batch_idx):
        prep_result = self._prepare_batch(batch)
        if prep_result is None:
            warnings.warn(f"Skipping training step {batch_idx}: Batch preparation failed fundamentally.")
            return None
        pixel_values, input_ids, attention_mask, labels = prep_result

        # Check if we have labels for this batch (essential for training)
        if labels is None or (self.mode == 'vqa' and input_ids is None):
             warnings.warn(f"Skipping training step {batch_idx}: Missing required inputs/labels for mode '{self.mode}'.")
             return None # Skip step

        # Perform forward pass to get loss
        try:
             outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
             loss = outputs.loss
        except Exception as e_fwd:
             warnings.warn(f"Error during training forward pass {batch_idx}: {e_fwd}")
             return torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss

        if loss is None:
             warnings.warn(f"Loss is None for training batch {batch_idx}. Check model output and label preparation.")
             return None # Skip if loss calculation failed

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # Re-enabled validation_step
    def validation_step(self, batch, batch_idx):
        prep_result = self._prepare_batch(batch)
        if prep_result is None:
             warnings.warn(f"Skipping validation step {batch_idx}: Batch preparation failed.")
             return None
        pixel_values, input_ids, attention_mask, labels = prep_result

        if labels is None or (self.mode == 'vqa' and input_ids is None):
             warnings.warn(f"Skipping validation step {batch_idx}: Missing required inputs/labels for mode '{self.mode}'.")
             return None

        try:
            outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        except Exception as e_fwd:
             warnings.warn(f"Error during validation forward pass {batch_idx}: {e_fwd}")
             return None # Skip logging if forward pass fails

        if loss is not None:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Optionally generate text and calculate metrics here (can be slow)
            # Or calculate metrics in validation_epoch_end / using a callback
            return loss
        else:
             warnings.warn(f"Loss is None for validation batch {batch_idx}.")
             return None


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
        try:
            # Calculate total steps (required for some schedulers)
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                 # estimated_stepping_batches includes train and possibly val batches, adjust if needed
                 self.total_training_steps = self.trainer.estimated_stepping_batches # Use trainer's estimate
                 print(f"Using trainer's estimated_stepping_batches: {self.total_training_steps}")
            else:
                 # Estimate manually if trainer attribute not available yet
                 warnings.warn("Trainer attribute 'estimated_stepping_batches' not found. Estimating total steps.")
                 if self.trainer.limit_train_batches:
                     # Handle limit_train_batches if it's a float (fraction) or int (number)
                     limit = self.trainer.limit_train_batches
                     if isinstance(limit, float):
                         num_batches = int(len(self.trainer.datamodule.train_dataloader()) * limit)
                     else: num_batches = int(limit)
                 else: num_batches = len(self.trainer.datamodule.train_dataloader())

                 self.total_training_steps = num_batches // getattr(self.trainer,'accumulate_grad_batches', 1) * self.trainer.max_epochs
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
    DATA_DIR_S6 = "./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" # Adjusted path
    ANNOTATIONS_PATH_S6 = "./dummy_brats_annotations.csv" # Adjusted dummy path name

    if not os.path.exists(DATA_DIR_S6):
         print(f"ERROR: BraTS data directory not found at '{DATA_DIR_S6}'.")
         print("Please download the dataset using KaggleHub and ensure the path is correct.")
         exit()
    if not os.path.exists(ANNOTATIONS_PATH_S6):
         # Create dummy annotations matching example logic from Stage 1
         print(f"Creating dummy annotations file: {ANNOTATIONS_PATH_S6}")
         example_patient_ids = [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR_S6, "BraTS20_Training_*"))[:20]]
         if not example_patient_ids: example_patient_ids = [f"BraTS20_Training_{i:03}" for i in range(1,21)]
         dummy_annot_data = [{'patient_id': p_id, 'question': f'Q_{p_id}', 'answer': f'A_{p_id}', 'report': f'R_{p_id}'} for p_id in example_patient_ids]
         pd.DataFrame(dummy_annot_data).to_csv(ANNOTATIONS_PATH_S6, index=False)
         print(f"Created dummy annotations for IDs: {example_patient_ids}")


    data_module_instance = None
    try:
        data_module_instance = NeuroReportDataModule(
            data_dir=DATA_DIR_S6,
            annotations_path=ANNOTATIONS_PATH_S6,
            batch_size=BATCH_SIZE,
            mode="report", # Set mode
            target_size=TARGET_SIZE,
            val_split=VAL_SPLIT, # Enable validation split
            test_split=TEST_SPLIT, # Enable test split
            num_workers=0,
            n_slices=NUM_SLICES_PER_SCAN
        )
        data_module_instance.prepare_data()
        data_module_instance.setup('fit') # Setup train and val
    except Exception as e_dm:
         print(f"Error creating DataModule: {e_dm}"); data_module_instance = None

    # --- Instantiate the main model ---
    if data_module_instance:
        try:
            # Use the component instances created earlier
            neuro_report_model_instance = NeuroReportModel(
                vision_encoder=vision_encoder_instance,
                slice_aggregator=slice_aggregator_instance,
                bridge=bridge_instance,
                language_decoder=language_decoder_instance,
                tokenizer=tokenizer_instance, # Pass tokenizer
                is_encoder_decoder_model=is_encoder_decoder, # Pass model type flag
                mode="report", # Should match DataModule mode
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                warmup_steps_ratio=WARMUP_STEPS_RATIO,
                max_label_length=MAX_LABEL_LENGTH
            )

            # --- Configure Trainer ---
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            # Re-enable checkpointing based on validation loss
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=MODEL_SAVE_PATH,
                filename='neuroreport-{epoch:02d}-{val_loss:.2f}',
                save_top_k=1,        # Save only the best model based on val_loss
                monitor='val_loss',  # Monitor validation loss
                mode='min',
                save_last=True       # Also save the last checkpoint
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            # Re-enable EarlyStopping
            early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

            # Determine precision based on QLoRA status (logic from Stage 5/6)
            precision_setting = '32-true' # Default
            if torch.cuda.is_available():
                 if hasattr(language_decoder_instance, 'use_lora') and language_decoder_instance.use_lora and hasattr(language_decoder_instance.model, 'quantization_config'): # Check if QLoRA active
                      precision_setting = '32-true' # Recommended for stability with bitsandbytes
                      print("Using QLoRA (4-bit): Trainer precision set to 32-true.")
                 elif torch.cuda.is_bf16_supported():
                      precision_setting = 'bf16-mixed'; print("Using bfloat16 mixed precision.")
                 else:
                      precision_setting = '16-mixed'; print("Using float16 mixed precision.")

            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto",
                precision=precision_setting,
                callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback], # Add EarlyStopping
                log_every_n_steps=10,
                gradient_clip_val=GRADIENT_CLIP_VAL,
                num_sanity_val_steps=2, # Run a couple of sanity checks on val data
                # limit_train_batches=0.1, # DEBUG: Use fraction of training data
                # limit_val_batches=0.1,   # DEBUG: Use fraction of val data
            )

            print(f"\nPyTorch Lightning Trainer configured:")
            print(f"  - Mode: {neuro_report_model_instance.mode}")
            print(f"  - Max Epochs: {NUM_EPOCHS}")
            print(f"  - Precision: {trainer.precision}")
            print(f"  - Checkpoint Path: {MODEL_SAVE_PATH}")
            print("\nStarting training (fit call commented out for example)...")

            # --- Start Training ---
            # Re-enable validation dataloader in fit()
            # trainer.fit(neuro_report_model_instance, datamodule=data_module_instance)
            print("\nTrainer.fit(...) call is commented out.")
            print("To run training, ensure valid DataLoaders, dependencies, and uncomment the trainer.fit line.")

        except Exception as e:
            import traceback
            print(f"\nError setting up or running training: {e}")
            traceback.print_exc()
    else:
        print("Skipping Trainer setup because DataModule could not be initialized.")

    print("\nStage 6: Training setup complete.\n")
