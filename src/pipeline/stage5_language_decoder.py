# ========== Stage 5: Language Decoder – Biomedical Text Generation ==========
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer,
    BitsAndBytesConfig, # For 4-bit quantization
    GenerationConfig, # More explicit control over generation
)
from peft import (
    LoraConfig, TaskType, # For QLoRA / LoRA
    get_peft_model, prepare_model_for_kbit_training, PeftModel
)
import warnings
import os
import psutil # To check available memory
import gc # Garbage collection

# --- Import configurations from previous stages ---
try:
    # Use absolute import if stages are in the same package, relative if scripts
    from .stage1_data_acquisition import BATCH_SIZE # Needed for example usage
    from .stage4_vision_language_bridge import LANGUAGE_MODEL_DIM as STAGE4_TARGET_DIM
    TARGET_LANGUAGE_MODEL_DIM = STAGE4_TARGET_DIM
    print(f"Successfully derived TARGET_LANGUAGE_MODEL_DIM from Stage 4: {TARGET_LANGUAGE_MODEL_DIM}")

except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from previous stages ({e_imp}). Using placeholder values for Stage 5.")
    BATCH_SIZE = 4 # Placeholder
    # Placeholder - MUST match the actual output dim of the bridge in Stage 4
    TARGET_LANGUAGE_MODEL_DIM = 768 # Example (e.g., ViT-Base projected to T5-base)
except NameError as e_name:
     warnings.warn(f"LANGUAGE_MODEL_DIM not found in stage4 ({e_name}). Check stage4 definition. Using placeholder.")
     BATCH_SIZE = 4 # Placeholder
     TARGET_LANGUAGE_MODEL_DIM = 768
except Exception as e_other:
     warnings.warn(f"Error importing config from previous stages ({e_other}). Using placeholder values.")
     BATCH_SIZE = 4 # Placeholder
     TARGET_LANGUAGE_MODEL_DIM = 768

# --- Configuration ---
# Choose a biomedical or general-purpose LM
LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Example Seq2Seq (Encoder-Decoder) ~900MB
# LANGUAGE_MODEL_NAME = 'microsoft/BioGPT-Base' # Example CausalLM (Decoder-Only) ~1.5GB
# LANGUAGE_MODEL_NAME = 'facebook/opt-350m' # Another CausalLM example ~700MB
# LANGUAGE_MODEL_NAME = "tiiuae/falcon-7b" # Large CausalLM example ~14GB+ (Requires significant GPU RAM)

# QLoRA / LoRA specific configuration
USE_QLORA = True # Enable/disable QLoRA (4-bit quantization + LoRA)
LORA_R = 16          # LoRA rank (higher rank -> more parameters, potentially better fit but slower)
LORA_ALPHA = 32      # LoRA alpha scaling (often 2*r)
LORA_DROPOUT = 0.05
# IMPORTANT: Adjust target_modules based on the specific model architecture!
# Use model inspection (print(model)) to find suitable linear/attention layer names.
# Common names for T5: ["q", "v"]
# Common names for OPT/GPT/BioGPT: ["q_proj", "v_proj"]
# Common names for Llama/Falcon: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] (check specific variant)
# Initialize with None or empty list, will be determined heuristically or requires manual setting
LORA_TARGET_MODULES = None # Example: ["q", "v"] for T5. If None, will try to infer.

# Generation parameters
MAX_GENERATION_LENGTH = 128 # Max *new* tokens for generated output
NUM_BEAMS = 4 # For beam search generation (1 for greedy)
DO_SAMPLE = False # Whether to use sampling (True) or deterministic (False, e.g., beam search/greedy)
TEMPERATURE = 0.7 # Softmax temperature for sampling (if DO_SAMPLE=True)
TOP_K = 50        # Top-k sampling (if DO_SAMPLE=True)
TOP_P = 0.95      # Nucleus sampling (if DO_SAMPLE=True)


# --- Helper Functions ---
def get_gpu_memory():
    """Returns available GPU memory in GB using nvidia-smi or torch."""
    if torch.cuda.is_available():
        try:
            # Try using nvidia-smi via subprocess (might be more reliable for free memory)
            result = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader').read()
            free_mem_mib = int(result.strip())
            return free_mem_mib / 1024
        except Exception:
            # Fallback to torch.cuda.mem_get_info() which gives free and total
            # free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
            # return free_mem_bytes / (1024**3)
            # Fallback 2: Get total and allocated, calculate free
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
                return total_mem - allocated_mem # Approximation
            except Exception:
                return 0 # If CUDA properties fail
    else:
        return 0

def get_cpu_memory():
    """Returns available CPU memory in GB."""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 0

# Check memory before deciding quantization etc.
available_gpu_gb = get_gpu_memory()
available_cpu_gb = get_cpu_memory()
print(f"Available GPU Memory: {available_gpu_gb:.2f} GB")
print(f"Available CPU Memory: {available_cpu_gb:.2f} GB")

# --- Setup Quantization and LoRA ---
bnb_config = None
peft_config = None
model_load_kwargs = {}
# device_map = "auto" # Can cause issues with requires_grad, handle manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine if we *can* use QLoRA based on CUDA availability
can_use_qlora = USE_QLORA and torch.cuda.is_available()

if can_use_qlora:
    print("Setting up QLoRA configuration...")
    # Check for bfloat16 support
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using compute dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Recommended type
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True, # Recommended for stability
    )
    model_load_kwargs["quantization_config"] = bnb_config
    # model_load_kwargs["device_map"] = device_map # Avoid device_map with manual PEFT wrapping
    model_load_kwargs["low_cpu_mem_usage"] = True # Try to reduce CPU RAM usage during loading

    # Determine model type for PEFT TaskType and default LORA_TARGET_MODULES if not set
    model_name_lower = LANGUAGE_MODEL_NAME.lower()
    if "t5" in model_name_lower or "bart" in model_name_lower:
        task_type = TaskType.SEQ_2_SEQ_LM
        default_targets = ["q", "v"]
    elif "gpt" in model_name_lower or "opt" in model_name_lower or "llama" in model_name_lower or "falcon" in model_name_lower or "mistral" in model_name_lower:
        task_type = TaskType.CAUSAL_LM
        default_targets = ["q_proj", "v_proj"] # Common starting point
        if "llama" in model_name_lower or "mistral" in model_name_lower:
             # Llama/Mistral often benefit from targeting more layers
             default_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "falcon" in model_name_lower:
             default_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        warnings.warn(f"Could not reliably determine PEFT TaskType or default LoRA targets for {LANGUAGE_MODEL_NAME}. Defaulting to CAUSAL_LM and ['q_proj', 'v_proj']. Adjust manually if needed.")
        task_type = TaskType.CAUSAL_LM
        default_targets = ["q_proj", "v_proj"]

    # Use manually set LORA_TARGET_MODULES if provided, otherwise use inferred defaults
    if LORA_TARGET_MODULES is None:
         LORA_TARGET_MODULES = default_targets
         print(f"Using inferred LoRA target modules: {LORA_TARGET_MODULES}")
    else:
        print(f"Using manually specified LoRA target modules: {LORA_TARGET_MODULES}")


    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", # Usually set to 'none' for LoRA
        task_type=task_type
    )
    print(f"PEFT Config: TaskType={task_type}, R={LORA_R}, Alpha={LORA_ALPHA}, Dropout={LORA_DROPOUT}, TargetModules={LORA_TARGET_MODULES}")
else:
    # If not using QLoRA, load model normally
    # Optionally use float16 if GPU supports it and has enough VRAM
    if device.type == 'cuda' and available_gpu_gb > 6: # Heuristic threshold for float16
        print("Attempting to load model in float16...")
        model_load_kwargs["torch_dtype"] = torch.float16
    pass

# --- Load Tokenizer ---
print(f"Loading Tokenizer: {LANGUAGE_MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    # Set padding token if not present (common for CausalLMs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer pad_token set to eos_token ({tokenizer.eos_token}).")
        else:
            # Add a standard pad token if no EOS token exists
            new_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            print(f"Added '{new_pad_token}' as pad_token.")
            # Important: Resize model embeddings if a new token was added
            # This needs to happen *before* applying PEFT if model is loaded first
            # We'll handle resizing after model loading for simplicity here.

except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer '{LANGUAGE_MODEL_NAME}': {e}")

# --- Load Model ---
print(f"Loading Language Model: {LANGUAGE_MODEL_NAME} {'with QLoRA' if can_use_qlora else ('with float16' if model_load_kwargs.get('torch_dtype') == torch.float16 else '')}")
language_model = None # Initialize to None
is_encoder_decoder = False # Initialize flag

try:
    # Determine model class based on name heuristic (adjust if needed)
    model_name_lower = LANGUAGE_MODEL_NAME.lower()
    if "t5" in model_name_lower or "bart" in model_name_lower:
        is_encoder_decoder = True
        model_class = AutoModelForSeq2SeqLM
    else:
        is_encoder_decoder = False
        model_class = AutoModelForCausalLM

    language_model = model_class.from_pretrained(
        LANGUAGE_MODEL_NAME,
        **model_load_kwargs # Pass quantization/dtype args
    )
    print(f"Model loaded. Type: {'Seq2Seq' if is_encoder_decoder else 'CausalLM'}")

    # Resize embeddings if pad token was added *after* loading the model
    if hasattr(tokenizer, 'new_pad_token_added') and tokenizer.new_pad_token_added:
         language_model.resize_token_embeddings(len(tokenizer))
         print(f"Resized model token embeddings to {len(tokenizer)} due to added pad token.")
         delattr(tokenizer, 'new_pad_token_added') # Clean up flag


    # --- Verify Model Dimension ---
    # Check if the loaded model's dimension matches the expected dimension from Stage 4
    try:
        actual_lm_dim = None
        if hasattr(language_model.config, 'hidden_size'):
             actual_lm_dim = language_model.config.hidden_size
        elif hasattr(language_model.config, 'd_model'): # Common in T5/BART
            actual_lm_dim = language_model.config.d_model
        elif hasattr(language_model.config, 'n_embd'): # Common in older GPT models
            actual_lm_dim = language_model.config.n_embd

        if actual_lm_dim is None:
             # Try to get from embedding layer if config is unclear
             embed_layer = None
             if hasattr(language_model, 'get_input_embeddings'):
                  embed_layer = language_model.get_input_embeddings()
             elif hasattr(language_model, 'transformer') and hasattr(language_model.transformer, 'wte'): # GPT-2 style
                  embed_layer = language_model.transformer.wte

             if embed_layer:
                  actual_lm_dim = embed_layer.embedding_dim
             else:
                 warnings.warn("Could not determine actual LM dimension from config or embeddings.")

        if actual_lm_dim and actual_lm_dim != TARGET_LANGUAGE_MODEL_DIM:
            warnings.warn(f"Potential Mismatch! Stage 4 Target LM Dim ({TARGET_LANGUAGE_MODEL_DIM}) != Actual Loaded LM Dim ({actual_lm_dim}). Check Stage 4 config or LM choice. Bridge projection might be incorrect.")
        elif actual_lm_dim:
            print(f"Loaded LM dimension ({actual_lm_dim}) matches target dimension ({TARGET_LANGUAGE_MODEL_DIM}).")

    except Exception as e_dim:
        warnings.warn(f"Could not verify language model dimension: {e_dim}")


except Exception as e:
    print(f"\nError loading model '{LANGUAGE_MODEL_NAME}': {e}")
    print("Check model name, internet connection, and available memory (especially GPU RAM).")
    print("Try using a smaller model or enabling QLoRA if memory is the issue.")
    import traceback
    traceback.print_exc()
    # Ensure model is None if loading failed
    language_model = None


# --- Apply PEFT (LoRA adapters) if model loaded successfully and QLoRA enabled ---
if language_model and can_use_qlora and peft_config:
    print("Preparing model for k-bit training (enables gradient checkpointing, useful even for inference setup)...")
    try:
        # Gradient checkpointing reduces memory usage but can slightly slow down inference/training.
        # Set use_gradient_checkpointing=False if optimizing purely for inference speed.
        language_model = prepare_model_for_kbit_training(language_model, use_gradient_checkpointing=False)
        print("Model prepared for k-bit training.")
    except Exception as e_prep:
         # Some architectures might not need this or have issues
         warnings.warn(f"Could not prepare model for k-bit training: {e_prep}. Proceeding without it.")


    print(f"Applying PEFT (LoRA) adapters...")
    try:
        language_model = get_peft_model(language_model, peft_config)
        print("PEFT model created successfully.")
        language_model.print_trainable_parameters() # Shows % of trainable params (should be small)
    except ValueError as e:
        warnings.warn(f"\nCould not apply PEFT. Check LORA_TARGET_MODULES: {LORA_TARGET_MODULES} are valid Linear layers in the base model.")
        print(f"ValueError: {e}")
        print("You can inspect the base model's layers using print(base_model).")
        warnings.warn("Proceeding with the base quantized model WITHOUT LoRA adapters.")
        # Indicate LoRA is not active
        can_use_qlora = False
        peft_config = None # Nullify peft_config as it wasn't applied
    except Exception as e:
         warnings.warn(f"An unexpected error occurred applying PEFT: {e}\nProceeding without LoRA adapters.")
         can_use_qlora = False
         peft_config = None


# Ensure model is on the correct device if not using device_map
if language_model and "device_map" not in model_load_kwargs:
    print(f"Moving model to device: {device}")
    try:
        language_model.to(device)
        print(f"Model successfully moved to {device}.")
    except Exception as e_move:
         warnings.warn(f"Could not move model to {device}: {e_move}. Model might remain on CPU or previous device.")
elif language_model and hasattr(language_model, 'device'):
    # With device_map='auto', the model is already on assigned devices.
    # Update device variable for tensor placement.
    inferred_device = language_model.device
    print(f"Model loaded with device_map. Primary device inferred as: {inferred_device}")
    device = inferred_device
elif language_model:
     # Fallback if device attribute not present after device_map
     try:
          inferred_device = next(language_model.parameters()).device
          print(f"Model loaded with device_map. Inferring device from first parameter: {inferred_device}")
          device = inferred_device
     except Exception as e_infer_param:
          warnings.warn(f"Could not infer device from model parameters after device_map: {e_infer_param}. Using configured device: {device}")



# --- Generation Function ---
def generate_text_from_embedding(model: PreTrainedModel | PeftModel,
                                 tokenizer: PreTrainedTokenizer,
                                 visual_embedding: torch.Tensor,
                                 input_text_prompt: list[str] | None = None,
                                 max_new_tokens: int = MAX_GENERATION_LENGTH,
                                 num_beams: int = NUM_BEAMS,
                                 do_sample: bool = DO_SAMPLE,
                                 temperature: float = TEMPERATURE,
                                 top_k: int = TOP_K,
                                 top_p: float = TOP_P,
                                 is_encoder_decoder_model: bool = is_encoder_decoder):
    """
    Generates text using the language model, conditioned on the visual embedding.

    Args:
        model: The loaded language model (potentially PEFT-wrapped).
        tokenizer: The corresponding tokenizer.
        visual_embedding (torch.Tensor): Conditioned visual embedding from Stage 4
                                          Shape: (BatchSize, TargetLanguageModelDimension)
        input_text_prompt (list[str], optional): List of input prompts (e.g., questions).
                                                If None, assumes unconditional generation from BOS.
        max_new_tokens (int): Max *new* tokens to generate (excluding prompt).
        num_beams (int): Number of beams for beam search (if do_sample=False).
        do_sample (bool): Use sampling if True, otherwise deterministic (False, e.g., beam search/greedy).
        temperature (float): Softmax temperature for sampling.
        top_k (int): Top-k sampling parameter.
        top_p (float): Nucleus sampling parameter.
        is_encoder_decoder_model (bool): Flag indicating model type.

    Returns:
        list[str]: List of generated text strings.
    """
    if model is None:
        warnings.warn("Model is None, cannot generate text.")
        return ["Model not loaded."] * visual_embedding.shape[0]

    model.eval() # Ensure model is in evaluation mode
    model_device = next(model.parameters()).device # Get device where model parameters reside

    batch_size = visual_embedding.shape[0]

    # --- Prepare inputs for the model ---
    prompt_length = 0 # Initialize prompt length
    if input_text_prompt:
        # Check batch size consistency
        if len(input_text_prompt) != batch_size:
             raise ValueError(f"Batch size mismatch: visual_embedding ({batch_size}) vs input_text_prompt ({len(input_text_prompt)})")
        inputs = tokenizer(input_text_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512) # Max length for input prompt
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device)
        prompt_length = input_ids.shape[1] # Length of the tokenized prompt
    else:
        # For unconditional generation (e.g., report), start with BOS token for CausalLM
        # Seq2Seq models often handle decoder start internally if `input_ids` isn't given to `generate`.
        if not is_encoder_decoder_model:
             # Ensure BOS token ID exists
             bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id # Fallback to EOS if BOS is missing
             if bos_token_id is None:
                 raise ValueError("Tokenizer must have a bos_token_id or eos_token_id for unconditional CausalLM generation.")
             input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(model_device)
             attention_mask = torch.ones_like(input_ids)
             prompt_length = 1
        else:
            # For Seq2Seq unconditional, let generate handle decoder start
            input_ids = None # Or tokenizer("", return_tensors="pt").input_ids.to(model_device) ? Check HF docs.
            attention_mask = None
            prompt_length = 0 # No explicit prompt tokens

    # --- Prepare visual features ---
    # Needs shape (BatchSize, SequenceLength=1, EmbeddingDim) for cross-attention
    encoder_hidden_states = visual_embedding.unsqueeze(1).to(model_device)


    # --- Setup GenerationConfig ---
    # More explicit way to control generation parameters
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0, # Temp doesn't matter if not sampling
        top_k=top_k if do_sample else None,            # Only use top_k if sampling
        top_p=top_p if do_sample else None,            # Only use top_p if sampling
        early_stopping=True if num_beams > 1 else False, # Stop when beams finish
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Force setting bos_token_id for Causal LMs if needed by generation logic (sometimes helps)
        bos_token_id=tokenizer.bos_token_id if not is_encoder_decoder_model else None,
        # Suppress specific warnings during generation if needed
        # suppress_warnings=True,
    )

    # --- Prepare model inputs based on type ---
    model_inputs = {}
    if is_encoder_decoder_model:
        # Pass visual features as encoder output for the decoder to attend to
        model_inputs["encoder_outputs"] = (encoder_hidden_states,) # Must be a tuple wrapping the tensor
        # The `input_ids` provided here are used as `decoder_input_ids` to start generation
        if input_ids is not None:
             model_inputs["input_ids"] = input_ids
             # Pass attention_mask if input_ids are provided (for decoder)
             model_inputs["attention_mask"] = attention_mask
        else:
             # If no input prompt, generate handles decoder start (usually pad token)
             # T5/FLAN-T5 might expect a single pad token as decoder input for unconditional gen
             # Let Huggingface `generate` handle this default.
             pass

    else: # Causal LM
        # Standard Causal LMs don't have a separate encoder input. Conditioning requires modifications.
        # Option B: Pass visual embeddings via `inputs_embeds`. Needs careful construction.
        # Here, we demonstrate passing standard inputs, acknowledging limitations.
        warnings.warn("Standard CausalLM generation with visual conditioning via 'encoder_hidden_states' might be ignored by the model. Architectural changes (e.g., custom embedding layer, multimodal models) are typically required for effective CausalLM visual conditioning.")

        # If we have starting input_ids (like BOS or prompt), pass them
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
            model_inputs["attention_mask"] = attention_mask

        # We *could* try passing visual features as `encoder_hidden_states` if the CausalLM has cross-attention
        # enabled (e.g., some modified architectures), but it's non-standard for typical GPT-like models.
        # For demonstration, we do NOT pass encoder_hidden_states here for standard Causal LMs.
        # model_inputs["encoder_hidden_states"] = encoder_hidden_states # Usually ignored


    # --- Generate Text ---
    print(f"Starting generation with config: {generation_config}")
    print(f"Model inputs keys: {list(model_inputs.keys())}")
    if "input_ids" in model_inputs and model_inputs["input_ids"] is not None:
        print(f"Input IDs shape: {model_inputs['input_ids'].shape}")
    if "encoder_outputs" in model_inputs:
        print(f"Encoder outputs (visual features) shape: {model_inputs['encoder_outputs'][0].shape}")


    output_sequences = None
    try:
        with torch.no_grad():
            output_sequences = model.generate(
                **model_inputs,
                generation_config=generation_config
            )
            print("Generation complete.")
            # print(f"Raw output sequences shape: {output_sequences.shape}") # Shape (BatchSize, GeneratedSeqLen)
    except Exception as e_gen:
        print(f"\nError during model.generate(): {e_gen}")
        print("Check model inputs, generation config, and model compatibility.")
        import traceback
        traceback.print_exc()
        # Create dummy output sequences if generation failed
        # Ensure the dummy sequence length is at least prompt_length + 1
        dummy_len = prompt_length + 1
        output_sequences = torch.full((batch_size, dummy_len), tokenizer.pad_token_id, device=model_device)


    # --- Decode generated sequences ---
    # For CausalLMs, remove prompt tokens from the beginning if a prompt was given
    if not is_encoder_decoder_model and prompt_length > 0:
         # Slice the output sequences to remove the prompt part
         actual_output_sequences = output_sequences[:, prompt_length:]
    else:
         # For Seq2Seq, the output doesn't include the input prompt
         # Also for CausalLM started with only BOS (prompt_length=1), we keep the full output
         actual_output_sequences = output_sequences

    #print(f"Output sequences after slicing prompt (if any): {actual_output_sequences.shape}")

    generated_texts = tokenizer.batch_decode(actual_output_sequences, skip_special_tokens=True)

    # Clean up memory
    del model_inputs, encoder_hidden_states
    if 'input_ids' in locals() and input_ids is not None: del input_ids
    if 'attention_mask' in locals() and attention_mask is not None: del attention_mask
    if output_sequences is not None: del output_sequences
    if 'actual_output_sequences' in locals() and actual_output_sequences is not None: del actual_output_sequences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect() # Force garbage collection

    return generated_texts

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 5 Example ---")

    if language_model is None:
        print("Language model failed to load. Skipping generation example.")
    else:
        # Assume conditioned_visual_embedding from Stage 4 exists (B, TargetLanguageModelDim)
        # Or create dummy data:
        # Use TARGET_LANGUAGE_MODEL_DIM which was derived or set as placeholder
        dummy_conditioned_embedding = torch.randn(BATCH_SIZE, TARGET_LANGUAGE_MODEL_DIM)
        print(f"Input dummy conditioned embedding shape: {dummy_conditioned_embedding.shape}")

        # Example VQA prompts
        vqa_prompts = [f"Describe the findings in scan number {i+1} based on the visual features." for i in range(BATCH_SIZE)]
        # Example Report Generation (no text prompt for seq2seq, maybe BOS for causal)
        report_prompts = None # Let generate handle start for seq2seq
        # If causal, you might give a generic starting prompt:
        # report_prompts = ["Findings:"] * BATCH_SIZE

        try:
            # --- Generate VQA answers ---
            print("\nGenerating VQA Example...")
            # Move embedding to the same device as the model
            dummy_conditioned_embedding_vqa = dummy_conditioned_embedding.to(device)

            vqa_answers = generate_text_from_embedding(
                language_model,
                tokenizer,
                dummy_conditioned_embedding_vqa,
                input_text_prompt=vqa_prompts,
                max_new_tokens=64, # Shorter answers for VQA
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                is_encoder_decoder_model=is_encoder_decoder
            )
            print("-" * 20)
            for i, answer in enumerate(vqa_answers):
                print(f"Prompt {i}: {vqa_prompts[i]}")
                print(f"Answer {i}: {answer}")
                print("-" * 10)
            del dummy_conditioned_embedding_vqa, vqa_answers # Clean up memory

            # --- Generate Reports ---
            print("\nGenerating Report Example...")
            # Move embedding to the same device as the model
            dummy_conditioned_embedding_rep = dummy_conditioned_embedding.to(device)

            reports = generate_text_from_embedding(
                language_model,
                tokenizer,
                dummy_conditioned_embedding_rep,
                input_text_prompt=report_prompts, # Use None or specific starting prompts
                max_new_tokens=MAX_GENERATION_LENGTH, # Longer for reports
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                is_encoder_decoder_model=is_encoder_decoder
            )
            print("-" * 20)
            for i, report in enumerate(reports):
                print(f"Generated Report {i}: {report}")
                print("-" * 10)
            del dummy_conditioned_embedding_rep, reports # Clean up memory


        except ImportError as ie:
            print(f"\nImportError: {ie}. Make sure necessary libraries (torch, transformers, peft, accelerate, bitsandbytes) are installed.")
        except Exception as e:
            import traceback
            print(f"\nError during text generation example: {e}")
            traceback.print_exc()

    # Optional: Clean up model and tokenizer from memory
    if 'language_model' in locals() and language_model is not None: del language_model
    if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Cleaned up model and tokenizer.")

    print("\nStage 5: Language model loading and text generation setup complete.\n")
