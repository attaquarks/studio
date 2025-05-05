# NeuroReport: Medical Imaging VQA & Report Generation Pipeline

This project implements a pipeline for Visual Question Answering (VQA) and report generation on medical MRI scans, specifically designed for the BraTS 2020 dataset.

## Overview

The pipeline consists of several stages:

1.  **Data Acquisition & Preprocessing**: Loads BraTS 2020 NIfTI files, handles multiple modalities (e.g., T1ce, FLAIR), preprocesses slices (normalization, resizing), and prepares data batches. Uses `kagglehub` to download the dataset.
2.  **Vision Encoder**: Extracts features from individual MRI slices using a Vision Transformer (ViT) model.
3.  **Slice Aggregation**: Combines slice features into a single scan-level embedding using methods like LSTM, GRU, Transformer, or mean pooling.
4.  **Vision-Language Bridge**: Projects the visual embedding to match the language model's expected dimension, enabling conditioning. Uses Cross-Attention.
5.  **Language Decoder**: Generates text (reports or VQA answers) using a biomedical language model (e.g., BioGPT, Flan-T5), conditioned on the visual features. Supports QLoRA for efficient fine-tuning.
6.  **Training Module**: Integrates all components into a PyTorch Lightning module for end-to-end training.
7.  **Evaluation**: (Currently Disabled) Placeholder for evaluating model performance using metrics like BLEU, ROUGE, METEOR, Accuracy.
8.  **Inference Demo**: (Currently Disabled) Placeholder for an interactive Gradio demo.
9.  **Training Script**: Main script to configure and run the training process.
10. **Main Entry Point**: Orchestrates the pipeline execution (currently focuses on training).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up Kaggle API Token:**
    *   Go to your Kaggle account settings (`https://www.kaggle.com/<your-username>/account`).
    *   Click "Create New API Token". This will download `kaggle.json`.
    *   Place the `kaggle.json` file in the expected location (usually `~/.kaggle/kaggle.json`).
    *   Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install torch torchvision torchaudio pytorch-lightning timm transformers peft bitsandbytes accelerate sentencepiece pandas scikit-learn nibabel opencv-python evaluate rouge_score sacrebleu nltk gradio kagglehub
    # If using METEOR, download NLTK data:
    # python -m nltk.downloader wordnet omw-1.4 punkt
    ```
    *Note: `bitsandbytes` might require specific build steps depending on your CUDA version.*

## Usage

The pipeline is primarily executed through `src/pipeline/stage10_main.py`, which uses `src/pipeline/stage9_training_script.py` to handle training.

1.  **Download Dataset (Handled Automatically):**
    The training script (`stage9`) will attempt to download the `awsaf49/brats2020-training-data` dataset using `kagglehub` if the specified `--data_dir` is not found. It will download to `./brats2020_kagglehub_download` by default. Ensure your Kaggle API token is set up.

2.  **Prepare Annotations (Optional but Recommended):**
    *   Create a CSV or JSON file containing annotations.
    *   It **must** have a `patient_id` column that matches the BraTS folder names (e.g., `BraTS20_Training_001`).
    *   For **report generation** (`--mode report`), include a `report` column with the target text.
    *   For **VQA** (`--mode vqa`), include `question` and `answer` columns.
    *   If no annotations file is provided (`--annotations_path None`), the model trains solely on the images (unsupervised pre-training or generation without explicit targets). The script will create a dummy annotations file for structure if none is found at the default path.

3.  **Run Training:**
    Execute the main script, providing necessary arguments.

    **Example (Report Generation Mode, using defaults):**
    ```bash
    python -m src.pipeline.stage10_main --data_dir ./brats2020_kagglehub_download/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --mode report --annotations_path ./path/to/your/report_annotations.csv --max_epochs 5 --batch_size 2 --accelerator gpu --devices 1 --checkpoint_dir ./brats_report_checkpoints
    ```

    **Example (VQA Mode, custom LM, QLoRA disabled):**
    ```bash
    python -m src.pipeline.stage10_main --data_dir ./brats2020_kagglehub_download/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --mode vqa --annotations_path ./path/to/your/vqa_annotations.csv --language_model_name google/flan-t5-base --use_4bit=false --use_lora=false --max_epochs 10 --batch_size 4 --learning_rate 5e-5 --checkpoint_dir ./brats_vqa_checkpoints
    ```

    **Key Arguments:**
    *   `--data_dir`: Path to the directory containing `BraTS20_Training_XXX` folders.
    *   `--annotations_path`: Path to your CSV/JSON annotations file (optional).
    *   `--mode`: `vqa` or `report`.
    *   `--max_epochs`: Number of training epochs.
    *   `--batch_size`: Adjust based on GPU memory.
    *   `--accelerator`, `--devices`: Configure hardware usage.
    *   `--checkpoint_dir`: Where to save model checkpoints.
    *   `--use_4bit`, `--use_lora`: Enable/disable QLoRA.
    *   See `src/pipeline/stage9_training_script.py` for all available arguments.

4.  **Inference/Evaluation (Currently Disabled):**
    The evaluation (Stage 7) and demo (Stage 8) components are disabled because the current configuration trains on the entire dataset without a validation/test split. To enable these:
    *   Modify `src/pipeline/stage1_data_acquisition.py` to implement data splitting.
    *   Re-enable and adapt the code in `src/pipeline/stage7_evaluation.py` and `src/pipeline/stage8_inference.py`.
    *   Adjust `src/pipeline/stage9_training_script.py` and `src/pipeline/stage10_main.py` to include evaluation/demo steps and arguments.

## Checkpoints

Model checkpoints will be saved in the directory specified by `--checkpoint_dir` during training. The best/last checkpoint can be used for continued training or inference (once enabled).
```
