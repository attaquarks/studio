# ========== Stage 5: Language Decoder – Biomedical Text Generation ==========
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer,
    BitsAndBytesConfig, # For 4-bit quantization
    GenerationConfig, # More explicit control over generation
)
from peft import (
    LoraConfig, TaskType, PeftModel, # For QLoRA / LoRA
    get_peft_model, prepare_model_for_kbit_training
)
import warnings
import os
import psutil # To check available memory
import gc # Garbage collection
from typing import Optional, List, Union

# --- Import configurations from previous stages ---
try:
    from .stage1_data_acquisition import BATCH_SIZE
    # Stage 4 now provides the cross-attention bridge, Stage 5 mainly needs the LM name/config
    # The TARGET_LANGUAGE_MODEL_DIM should be implicitly defined by the chosen LM in this stage.
except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from previous stages ({e_imp}). Using placeholder values for Stage 5.")
    BATCH_SIZE = 4


# --- Configuration for Stage 5 ---
# Choose a biomedical or general-purpose LM
# LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Example Seq2Seq (Encoder-Decoder) ~900MB
LANGUAGE_MODEL_NAME = 'microsoft/BioGPT-Large' # Example CausalLM (Decoder-Only) ~1.5GB
# LANGUAGE_MODEL_NAME = 'facebook/opt-350m' # Another CausalLM example ~700MB
# LANGUAGE_MODEL_NAME = "tiiuae/falcon-7b" # Large CausalLM example ~14GB+ (Requires significant GPU RAM)

# Quantization and LoRA configuration
USE_4BIT = True # Enable/disable 4-bit quantization (via bitsandbytes)
USE_LORA = True # Enable/disable LoRA adapters (requires USE_4BIT=True for QLoRA)

# LoRA specific configuration (only relevant if USE_LORA=True)
LORA_R = 16          # LoRA rank
LORA_ALPHA = 32      # LoRA alpha scaling
LORA_DROPOUT = 0.05
# Target modules for LoRA (can be model-specific)
# If None, will attempt to infer based on model type.
# Common for BioGPT/GPT/OPT: ["q_proj", "v_proj", "k_proj", "out_proj"]
# Common for T5: ["q", "v"]
LORA_TARGET_MODULES: Optional[List[str]] = None # Example: ["q_proj", "v_proj"]

# Generation parameters (can be overridden during inference)
MAX_GENERATION_LENGTH = 256 # Max *new* tokens for generated output
NUM_BEAMS = 4 # For beam search generation (1 for greedy)
NO_REPEAT_NGRAM_SIZE = 3 # Prevent repeating n-grams
EARLY_STOPPING = True # Stop beam search early if possible

# --- Helper Functions ---
def get_gpu_memory():
    """Returns available GPU memory in GB using nvidia-smi or torch."""
    if torch.cuda.is_available():
        try:
            result = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader').read()
            free_mem_mib = int(result.strip())
            return free_mem_mib / 1024
        except Exception:
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
                return total_mem - allocated_mem
            except Exception: return 0
    return 0

def get_cpu_memory():
    """Returns available CPU memory in GB."""
    try: return psutil.virtual_memory().available / (1024**3)
    except Exception: return 0

# Check memory
available_gpu_gb = get_gpu_memory()
available_cpu_gb = get_cpu_memory()
print(f"Available Memory: GPU={available_gpu_gb:.2f} GB, CPU={available_cpu_gb:.2f} GB")

# --- Determine Model Type and Task ---
model_name_lower = LANGUAGE_MODEL_NAME.lower()
if "t5" in model_name_lower or "bart" in model_name_lower:
    model_type = "seq2seq"
    model_class = AutoModelForSeq2SeqLM
    peft_task_type = TaskType.SEQ_2_SEQ_LM
    default_lora_targets = ["q", "v"]
elif "gpt" in model_name_lower or "opt" in model_name_lower or "llama" in model_name_lower or "falcon" in model_name_lower or "mistral" in model_name_lower:
    model_type = "causal_lm"
    model_class = AutoModelForCausalLM
    peft_task_type = TaskType.CAUSAL_LM
    # Default targets vary, provide common ones, adjust LORA_TARGET_MODULES if needed
    default_lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Generous defaults for Llama/Mistral like
    if "falcon" in model_name_lower:
         default_lora_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "opt" in model_name_lower or "gpt" in model_name_lower:
        default_lora_targets = ["q_proj", "v_proj"] # Simpler defaults
else:
    warnings.warn(f"Could not reliably determine model type for {LANGUAGE_MODEL_NAME}. Assuming CausalLM. Adjust 'model_type' if incorrect.")
    model_type = "causal_lm"
    model_class = AutoModelForCausalLM
    peft_task_type = TaskType.CAUSAL_LM
    default_lora_targets = ["q_proj", "v_proj"]

# Use manually specified LORA_TARGET_MODULES or inferred defaults
if LORA_TARGET_MODULES is None:
    LORA_TARGET_MODULES = default_lora_targets
    print(f"Using inferred LoRA target modules for {model_type}: {LORA_TARGET_MODULES}")
else:
     print(f"Using specified LoRA target modules: {LORA_TARGET_MODULES}")

# --- Setup Quantization and LoRA Configs ---
bnb_config = None
peft_config = None
model_load_kwargs = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

can_use_qlora = USE_4BIT and USE_LORA and torch.cuda.is_available()
can_use_quant = USE_4BIT and not USE_LORA and torch.cuda.is_available()

if can_use_qlora or can_use_quant:
    print(f"Setting up 4-bit quantization{' (QLoRA enabled)' if can_use_qlora else ' (LoRA disabled)'}...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using compute dtype: {compute_dtype}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model_load_kwargs["quantization_config"] = bnb_config
    model_load_kwargs["low_cpu_mem_usage"] = True

if can_use_qlora: # Setup PEFT config only if both 4bit and LoRA are enabled
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=peft_task_type
    )
    print(f"PEFT Config: TaskType={peft_task_type}, R={LORA_R}, Alpha={LORA_ALPHA}, Dropout={LORA_DROPOUT}, Targets={LORA_TARGET_MODULES}")
elif USE_LORA and not USE_4BIT:
     warnings.warn("LoRA is enabled (USE_LORA=True) but 4-bit quantization is disabled (USE_4BIT=False). Standard LoRA requires manual layer wrapping or different setup. Skipping LoRA application.")
     USE_LORA = False # Disable LoRA if not using QLoRA setup for simplicity here
else:
     USE_LORA = False # Ensure LoRA flag is False if not applicable


# --- Language Decoder Class ---
class LanguageDecoder(nn.Module):
    """Language decoder for biomedical text generation, potentially with QLoRA."""

    def __init__(self,
                 model_name: str = LANGUAGE_MODEL_NAME,
                 model_type: str = model_type, # Use determined type
                 use_lora: bool = USE_LORA,   # Use determined flag
                 **kwargs): # Absorb other potential args from NeuroReport init
        """
        Initialize the language decoder.

        Args:
            model_name: Name of the pretrained language model.
            model_type: Type ('causal_lm' or 'seq2seq').
            use_lora: Whether LoRA adapters are configured and applied.
            **kwargs: Additional arguments passed, including model_load_kwargs and peft_config.
        """
        super().__init__()

        self.model_name = model_name
        self.model_type = model_type
        self.use_lora = use_lora # Store whether LoRA is active

        # Load tokenizer
        print(f"Loading Tokenizer: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # Add special tokens (consider adding task-specific tokens like <img>, <QUESTION>)
            special_tokens = ["<img>", "</img>"] # Example image placeholders
            num_added = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
            if num_added > 0: print(f"Added {num_added} special tokens: {special_tokens}")

            # Set padding token if necessary
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print("Added '[PAD]' as pad_token.")
                    # Need to resize embeddings later if '[PAD]' was newly added

        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{self.model_name}': {e}")

        # Load model
        print(f"Loading Language Model: {self.model_name} (Type: {self.model_type})")
        print(f"  Quantization: {'4-bit' if bnb_config else 'None'}")
        print(f"  LoRA: {'Enabled' if self.use_lora and peft_config else 'Disabled'}")
        try:
            self.model = model_class.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **model_load_kwargs # Contains quantization config if enabled
            )
            print("Base model loaded.")

            # Resize embeddings if tokenizer added new tokens AFTER initial load
            if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
                print(f"Resizing model token embeddings from {self.model.get_input_embeddings().weight.shape[0]} to {len(self.tokenizer)}.")
                self.model.resize_token_embeddings(len(self.tokenizer))

        except Exception as e:
            print(f"\nError loading base model '{self.model_name}': {e}")
            print("Check model name, HF cache, internet, memory.")
            raise

        # Apply PEFT (LoRA) if configured
        if self.use_lora and peft_config:
            print("Applying PEFT (LoRA) adapters...")
            try:
                # Prepare model for k-bit training if using quantization (required for LoRA+4bit)
                if bnb_config:
                     print("Preparing model for k-bit training...")
                     self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False) # Set True only if training LoRA adapters

                self.model = get_peft_model(self.model, peft_config)
                print("PEFT model created successfully.")
                self.model.print_trainable_parameters()
            except ValueError as e_peft:
                warnings.warn(f"\nCould not apply PEFT. Check LORA_TARGET_MODULES: {LORA_TARGET_MODULES}. Error: {e_peft}\nProceeding with base {'quantized ' if bnb_config else ''}model.")
                self.use_lora = False # Disable LoRA flag as it wasn't applied
            except Exception as e_peft:
                 warnings.warn(f"An unexpected error occurred applying PEFT: {e_peft}\nProceeding without LoRA adapters.")
                 self.use_lora = False


        # Get model dimension
        self.model_dim = self.model.config.hidden_size
        print(f"Language model final dimension: {self.model_dim}")

        # Move model to device if not using device_map
        if "device_map" not in model_load_kwargs:
             self.model.to(device)
             print(f"Model moved to device: {device}")


    def get_model_dim(self) -> int:
        """Get the hidden dimension size of the language model."""
        return self.model_dim

    def prepare_inputs(self, input_text: Union[str, List[str]], max_length: int = 512, device=None) -> dict:
        """Tokenize input text for the language model."""
        if device is None:
             device = next(self.model.parameters()).device # Use model's device by default
        inputs = self.tokenizer(
            input_text,
            padding="longest", # Pad to longest in batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in inputs.items()} # Move tensors to device

    def forward(self, **kwargs):
        """
        Forward pass through the underlying language model.
        Accepts standard Hugging Face model arguments.
        """
        return self.model(**kwargs)

    def generate(self,
                 input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 encoder_hidden_states: Optional[torch.Tensor] = None, # Output from Stage 4 Bridge
                 max_new_tokens: int = MAX_GENERATION_LENGTH,
                 num_beams: int = NUM_BEAMS,
                 no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
                 early_stopping: bool = EARLY_STOPPING,
                 **generation_kwargs): # Allow passing other generation args
        """
        Generate text sequences using the language model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs. Required for CausalLM or as decoder start for Seq2Seq.
            attention_mask (torch.Tensor, optional): Attention mask for input_ids.
            encoder_hidden_states (torch.Tensor, optional): Output from the vision bridge (Stage 4),
                                                             used as context for the decoder.
                                                             Shape: [B, L_vis, D_lm] (L_vis often 1).
            max_new_tokens (int): Max number of *new* tokens to generate.
            num_beams (int): Number of beams for beam search.
            no_repeat_ngram_size (int): Size of n-grams to avoid repeating.
            early_stopping (bool): Whether to stop beam search early.
            **generation_kwargs: Additional arguments for the `generate` method.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        model_to_generate = self.model
        model_device = next(model_to_generate.parameters()).device

        # Prepare generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_kwargs # Incorporate additional args
        )

        # Prepare inputs for the generate method
        gen_inputs = {}
        if input_ids is not None:
             gen_inputs["input_ids"] = input_ids.to(model_device)
        if attention_mask is not None:
             gen_inputs["attention_mask"] = attention_mask.to(model_device)

        # Pass encoder_hidden_states if model is Seq2Seq or a CausalLM designed for cross-attention
        if self.model_type == "seq2seq" and encoder_hidden_states is not None:
             gen_inputs["encoder_outputs"] = (encoder_hidden_states.to(model_device),) # Must be tuple
        elif self.model_type == "causal_lm" and encoder_hidden_states is not None:
             # Passing encoder_hidden_states to standard CausalLMs might be ignored.
             # Needs model architecture modification (e.g., adding cross-attention layers).
             # Or use `inputs_embeds` for prefix-style conditioning.
             # For this structure, we pass it, but warn if it's likely standard CausalLM.
             if not hasattr(self.model.config, 'add_cross_attention') or not self.model.config.add_cross_attention:
                  warnings.warn("Passing 'encoder_hidden_states' to a standard CausalLM. This might be ignored. True visual conditioning often requires architecture changes or using 'inputs_embeds'.")
             # Some Causal models might accept it via 'encoder_hidden_states' kwarg if modified
             gen_inputs["encoder_hidden_states"] = encoder_hidden_states.to(model_device)


        print(f"Generating text with config: {generation_config}")
        # print(f"Generation input keys: {list(gen_inputs.keys())}")
        # if "input_ids" in gen_inputs: print(f"  input_ids shape: {gen_inputs['input_ids'].shape}")
        # if "encoder_outputs" in gen_inputs: print(f"  encoder_outputs shape: {gen_inputs['encoder_outputs'][0].shape}")
        # if "encoder_hidden_states" in gen_inputs: print(f"  encoder_hidden_states shape: {gen_inputs['encoder_hidden_states'].shape}")


        with torch.no_grad():
            output_sequences = model_to_generate.generate(
                **gen_inputs,
                generation_config=generation_config
            )

        return output_sequences

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

# --- Instantiate once to export LM dimension ---
try:
     # Minimal load just to get config if possible
     _temp_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
     _temp_config = model_class.config_class.from_pretrained(LANGUAGE_MODEL_NAME)
     LANGUAGE_MODEL_DIM = _temp_config.hidden_size
     print(f"Stage 5 LANGUAGE_MODEL_DIM determined: {LANGUAGE_MODEL_DIM}")
     del _temp_tokenizer, _temp_config
except Exception as e:
     warnings.warn(f"Could not determine LANGUAGE_MODEL_DIM dynamically from config: {e}. Using placeholder 768.")
     LANGUAGE_MODEL_DIM = 768 # Fallback


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Stage 5 Example ---")

    # Instantiate the decoder (will load model, tokenizer, apply QLoRA if enabled)
    try:
        language_decoder = LanguageDecoder()
        lm_device = next(language_decoder.model.parameters()).device
        print(f"LanguageDecoder initialized on device: {lm_device}")

        # --- Dummy Data for Generation Example ---
        # Assume conditioned_visual_embedding from Stage 4 exists (B, D_lm)
        # Create dummy data using the determined LANGUAGE_MODEL_DIM
        dummy_conditioned_embedding = torch.randn(BATCH_SIZE, LANGUAGE_MODEL_DIM).to(lm_device)
        # Reshape for encoder_hidden_states: [B, L_vis, D_lm] (L_vis=1)
        dummy_encoder_hidden = dummy_conditioned_embedding.unsqueeze(1)

        # Example VQA prompts
        vqa_prompts = [f"question: What abnormalities are seen near slice {i}?" for i in range(BATCH_SIZE)]
        # Example Report Generation prefix (can be empty for unconditional if model handles it)
        report_prompts = ["Report findings:"] * BATCH_SIZE # Example prefix

        # --- Generate VQA answers ---
        print("\nGenerating VQA Example...")
        vqa_inputs = language_decoder.prepare_inputs(vqa_prompts)
        generated_vqa_ids = language_decoder.generate(
            input_ids=vqa_inputs['input_ids'],
            attention_mask=vqa_inputs['attention_mask'],
            encoder_hidden_states=dummy_encoder_hidden, # Provide visual context
            max_new_tokens=64 # Shorter answers for VQA
        )
        vqa_answers = language_decoder.decode(generated_vqa_ids)
        print("-" * 20)
        for i, answer in enumerate(vqa_answers):
            print(f"Prompt {i}: {vqa_prompts[i]}")
            print(f"Answer {i}: {answer}")
            print("-" * 10)

        # --- Generate Reports ---
        print("\nGenerating Report Example...")
        report_inputs = language_decoder.prepare_inputs(report_prompts) # Tokenize prefix
        generated_report_ids = language_decoder.generate(
            input_ids=report_inputs['input_ids'],
            attention_mask=report_inputs['attention_mask'],
            encoder_hidden_states=dummy_encoder_hidden, # Provide visual context
            max_new_tokens=MAX_GENERATION_LENGTH # Longer for reports
        )
        reports = language_decoder.decode(generated_report_ids)
        print("-" * 20)
        for i, report in enumerate(reports):
            print(f"Prefix {i}: {report_prompts[i]}")
            print(f"Generated Report {i}: {report}")
            print("-" * 10)

        del language_decoder # Clean up model from memory if example run

    except ImportError as ie:
        print(f"\nImportError: {ie}. Missing libraries? (transformers, peft, accelerate, bitsandbytes)")
    except Exception as e:
        import traceback
        print(f"\nError during LanguageDecoder example: {e}")
        traceback.print_exc()

    # Optional: Further cleanup
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    print("\nStage 5: Language model loading and text generation setup complete.\n")
