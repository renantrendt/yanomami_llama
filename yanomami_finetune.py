#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Llama models on Yanomami language data.
This script uses the installed llama-cookbook package.

Usage:
    python yanomami_finetune.py --phase=1 --model_name="meta-llama/Meta-Llama-3.1-8B" --tokenizer_path="./fixed_tokenizer"
    python yanomami_finetune.py --phase=2 --model_name="meta-llama/Meta-Llama-3.1-8B" --tokenizer_path="./fixed_tokenizer"
"""

import argparse
import os
import logging
import sys
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable

import torch
from transformers import set_seed

# We'll import specific modules from the installed llama-cookbook package when needed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"finetune_yanomami.log")
    ]
)
logger = logging.getLogger(__name__)

# Define configuration classes here to avoid import issues
@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer_name: Optional[str] = None
    dataset_path: str = "./"
    phase: int = 1
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_samples: int = None  # Limit the number of samples for training
    context_length: int = 4096
    use_peft: bool = False
    lora_r: int = 128
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
    ])
    use_fp16: bool = False
    enable_fsdp: bool = False
    run_validation: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 100
    seed: int = 42
    data_seed: int = 42

    # Hardcoded file paths
    phase1_file: str = "formatted_data/phase1_data.txt"
    phase2_file: str = "formatted_data/phase2_data.txt"

@dataclass
class FSDPConfig:
    pure_bf16: bool = False
    optimizer: str = "adamw"
    fsdp_activation_checkpointing: bool = True

@dataclass
class QuantizationConfig:
    bits: int = 8 
    group_size: int = 128
    double_quant: bool = True

@dataclass
class WandbConfig:
    project: str = "llama-yanomami"
    run_name: str = "yanomami-finetune"
    watch_model: bool = False
    save_model: bool = False

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama models on Yanomami language data")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Training phase: 1 for translation-based training, 2 for bilingual next token prediction"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Name or path of the model to fine-tune"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./fixed_tokenizer",
        help="Path to the extended tokenizer"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./formatted_data",
        help="Path to the formatted training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=1000,
        help="Validation set size"
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Whether to use PEFT for fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples for training (e.g., 200 for ~200 words)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama-yanomami",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="W&B run name"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Whether to use FP16 precision"
    )
    parser.add_argument(
        "--enable_fsdp",
        action="store_true",
        help="Whether to enable FSDP for distributed training"
    )
    return parser.parse_args()

def preprocess_dataset(dataset, tokenizer, max_length=4096, is_train=True):
    """
    Preprocess the dataset for training, ensuring proper padding and truncation.
    This addresses the data formatting issue mentioned in the error message.
    """
    logger.info(f"Preprocessing dataset with {len(dataset)} examples")
    
    def tokenize_function(examples):
        # Handle different data formats based on the dataset structure
        if "text" in examples:
            # Standard text format
            texts = examples["text"]
        elif "input" in examples and "output" in examples:
            # Input-output format (common in instruction tuning)
            inputs = examples["input"]
            outputs = examples["output"]
            if isinstance(inputs, list) and isinstance(outputs, list):
                texts = [f"{i}\n\n{o}" for i, o in zip(inputs, outputs)]
            else:
                texts = f"{inputs}\n\n{outputs}"
        else:
            # Try to find any text-like field
            text_fields = [k for k in examples.keys() if any(t in k.lower() for t in ["text", "content", "data"])]
            if text_fields:
                texts = examples[text_fields[0]]
            else:
                # Fallback: convert the entire example to a string
                texts = str(examples)
        
        # Ensure texts is properly formatted
        if isinstance(texts, list):
            # Handle case where text is a list of strings
            texts = [t if isinstance(t, str) else str(t) for t in texts]
        elif not isinstance(texts, str):
            # Convert to string if not already a string
            texts = str(texts)
            
        # Log the first example to help with debugging
        if isinstance(texts, list) and len(texts) > 0:
            logger.info(f"Sample text (first 100 chars): {texts[0][:100]}...")
        elif isinstance(texts, str):
            logger.info(f"Sample text (first 100 chars): {texts[:100]}...")
            
        # Tokenize with proper padding and truncation
        try:
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            logger.error(f"Text type: {type(texts)}")
            if isinstance(texts, list):
                logger.error(f"First text item type: {type(texts[0]) if texts else 'empty list'}")
            raise
        
        # For causal language modeling, we need input_ids and labels
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        
        # For training, we also set labels equal to input_ids for causal LM
        if is_train:
            result["labels"] = tokenized["input_ids"].clone()
            
        return result
    
    # Apply the tokenization function to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    
    logger.info(f"Preprocessed dataset has {len(tokenized_dataset)} examples")
    return tokenized_dataset

def setup_train_config(args):
    """
    Set up the training configuration based on the command-line arguments.
    """
    # Directly set the data path based on the phase
    if args.phase == 1:
        data_path = "phase1_data.txt"
    else:
        data_path = "phase2_data.txt"
    
    # Set up the training configuration
    train_config = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_path,
        dataset_path=data_path,
        phase=args.phase,
        output_dir=os.path.join(args.output_dir, f"phase{args.phase}"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size if args.batch_size > args.micro_batch_size else 1,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.1,  # As per OpenHathi parameters
        warmup_ratio=0.1,  # As per OpenHathi parameters
        lr_scheduler_type="cosine",
        context_length=args.cutoff_len,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        use_fp16=args.use_fp16,
        enable_fsdp=args.enable_fsdp,
        run_validation=True,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        eval_steps=100,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # Set up the FSDP configuration if enabled
    fsdp_config = FSDPConfig(
        pure_bf16=False,
        optimizer="adamw",
        fsdp_activation_checkpointing=True,
    )
    
    # Set up the quantization configuration
    quant_config = QuantizationConfig()
    
    # Set up the W&B configuration if enabled
    wandb_config = WandbConfig(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"yanomami-phase{args.phase}",
        watch_model=False,
        save_model=False,
    ) if args.use_wandb else None
    
    return train_config, fsdp_config, quant_config, wandb_config

def main_cli():
    """
    Main function to set up and run the fine-tuning process.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up the training configuration
    train_config, fsdp_config, quant_config, wandb_config = setup_train_config(args)
    
    # Log the configuration
    logger.info(f"Training configuration: {asdict(train_config)}")
    if args.enable_fsdp:
        logger.info(f"FSDP configuration: {asdict(fsdp_config)}")
    if args.use_wandb:
        logger.info(f"W&B configuration: {asdict(wandb_config)}")
    
    # Import necessary functions from the local llama_cookbook.py file
    try:
        logger.info("Importing required modules...")
        
        # Import from the local llama_cookbook.py file
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current script directory: {current_dir}")
        
        # Check if llama_cookbook.py exists in the current directory
        cookbook_path = os.path.join(current_dir, 'llama_cookbook.py')
        if os.path.exists(cookbook_path):
            logger.info(f"Found local llama_cookbook.py at {cookbook_path}")
            
            # Import the main function directly from the local file
            import importlib.util
            spec = importlib.util.spec_from_file_location("llama_cookbook", cookbook_path)
            llama_cookbook_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llama_cookbook_module)
            
            # Get the main function from the local module
            llama_main = llama_cookbook_module.main
            logger.info("Successfully imported main function from local llama_cookbook.py")
        else:
            logger.error(f"Local llama_cookbook.py not found at {cookbook_path}")
            raise ImportError(f"Local llama_cookbook.py not found at {cookbook_path}")
        
        # Use datasets library directly instead of llama_cookbook.data
        from datasets import load_dataset, load_from_disk
        logger.info("Successfully imported datasets module.")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during module import: {str(e)}")
        sys.exit(1)
    
    # Check if the preprocessed dataset already exists
    preprocessed_path = os.path.join(os.path.dirname(train_config.dataset_path), "preprocessed")
    logger.info(f"Checking for preprocessed dataset at {preprocessed_path}")
    
    # Debug: Check if the data files exist
    logger.info(f"Checking if phase1 file exists: {os.path.exists(train_config.phase1_file)}")
    logger.info(f"Checking if phase2 file exists: {os.path.exists(train_config.phase2_file)}")
    
    # Debug: Print the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    
    if os.path.exists(preprocessed_path):
        logger.info(f"Preprocessed dataset already exists at {preprocessed_path}. Skipping processing.")
        train_config.dataset_path = preprocessed_path
        logger.info(f"Updated dataset_path to {train_config.dataset_path}")
        
        # Add detailed logging to track execution flow
        logger.info("========== PROCEEDING TO FINE-TUNING PREPARATION ==========")
        logger.info(f"Current train_config: {vars(train_config)}")
        logger.info(f"Current FSDP config: {vars(fsdp_config) if fsdp_config else 'None'}")
        logger.info(f"Current quant config: {vars(quant_config) if quant_config else 'None'}")
        logger.info(f"Current wandb config: {vars(wandb_config) if wandb_config else 'None'}")
        
        # Check for GPU availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            logger.warning("CUDA is not available. Training will be slow on CPU.")
            
        # Check if the formatted_data directory exists
        formatted_data_dir = os.path.join(os.path.dirname(train_config.dataset_path), "formatted_data")
        logger.info(f"Checking if formatted_data directory exists at {formatted_data_dir}")
        if os.path.exists(formatted_data_dir):
            logger.info(f"formatted_data directory exists. Contents: {os.listdir(formatted_data_dir)}")
        else:
            logger.warning(f"formatted_data directory does not exist at {formatted_data_dir}. Creating it...")
            try:
                os.makedirs(formatted_data_dir, exist_ok=True)
                logger.info(f"Created formatted_data directory at {formatted_data_dir}")
            except Exception as e:
                logger.error(f"Failed to create formatted_data directory: {str(e)}")
                
        # Check if the train.json file exists
        train_json_path = os.path.join(formatted_data_dir, "train.json")
        logger.info(f"Checking if train.json exists at {train_json_path}")
        if os.path.exists(train_json_path):
            logger.info(f"train.json exists. Size: {os.path.getsize(train_json_path)} bytes")
            try:
                with open(train_json_path, 'r', encoding='utf-8') as f:
                    num_lines = sum(1 for _ in f)
                logger.info(f"train.json contains {num_lines} lines")
                
                # Check if the required JSONL files exist
                train_jsonl = os.path.join(formatted_data_dir, "train.jsonl")
                val_jsonl = os.path.join(formatted_data_dir, "val.jsonl")
                test_jsonl = os.path.join(formatted_data_dir, "test.jsonl")
                
                logger.info(f"Checking for required JSONL files:\n  - train.jsonl exists: {os.path.exists(train_jsonl)}\n  - val.jsonl exists: {os.path.exists(val_jsonl)}\n  - test.jsonl exists: {os.path.exists(test_jsonl)}")
                
                # If the JSONL files don't exist, convert from train.json
                if not (os.path.exists(train_jsonl) and os.path.exists(val_jsonl) and os.path.exists(test_jsonl)):
                    logger.info("Converting train.json to the required JSONL format...")
                    
                    # Read the data from train.json
                    with open(train_json_path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f]
                    
                    # Split the data into train, validation, and test sets (80/10/10 split)
                    import random
                    random.seed(train_config.seed)  # For reproducibility
                    random.shuffle(data)
                    
                    # Limit the dataset size if max_samples is specified
                    if train_config.max_samples is not None:
                        logger.info(f"Limiting dataset to {train_config.max_samples} samples as specified")
                        data = data[:train_config.max_samples]
                        logger.info(f"Dataset limited to {len(data)} samples (approximately {len(data) * 5} words on average)")
                    
                    total_samples = len(data)
                    train_size = int(0.8 * total_samples)
                    val_size = int(0.1 * total_samples)
                    
                    train_data = data[:train_size]
                    val_data = data[train_size:train_size + val_size]
                    test_data = data[train_size + val_size:]
                    
                    logger.info(f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test samples")
                    
                    # Write the JSONL files
                    def write_jsonl(data_list, file_path):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            for item in data_list:
                                f.write(json.dumps(item) + '\n')
                        logger.info(f"Created {file_path} with {len(data_list)} entries")
                    
                    write_jsonl(train_data, train_jsonl)
                    write_jsonl(val_data, val_jsonl)
                    write_jsonl(test_data, test_jsonl)
                    logger.info("Successfully converted train.json to the required JSONL format")
                else:
                    logger.info("All required JSONL files already exist")
            except Exception as e:
                logger.error(f"Error processing train.json: {str(e)}")
        else:
            logger.warning(f"train.json does not exist at {train_json_path}")
            
        # Skip the execution flow check and proceed directly to fine-tuning
        logger.info("========== PROCEEDING DIRECTLY TO FINE-TUNING ==========")
        
        # Set the dataset path to the formatted_data directory
        train_config.dataset_path = formatted_data_dir
        logger.info(f"Set dataset_path to {train_config.dataset_path}")
        
        # Call the main function from llama_cookbook
        logger.info(f"Current train_config: {train_config.__dict__}")
        logger.info(f"Current FSDP config: {fsdp_config.__dict__}")
        logger.info(f"Current quant config: {quant_config.__dict__}")
        logger.info(f"Current wandb config: {wandb_config}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        else:
            logger.warning("CUDA is not available. Training will be slow.")
        
        # Import the main function from llama_cookbook
        try:
            # Set up Hugging Face token for authentication
            # os is already imported at the top of the file
            from huggingface_hub import login
            
            # Check if HF_TOKEN is set in environment variables
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                logger.warning("HF_TOKEN environment variable not found. Please set it to access gated models.")
                logger.info("You can set it using: export HF_TOKEN=your_token_here")
                # Ask for token if not in environment
                import getpass
                hf_token = getpass.getpass("Enter your Hugging Face token: ")
            
            # Login to Hugging Face
            logger.info("Logging in to Hugging Face...")
            login(token=hf_token)
            logger.info("Successfully logged in to Hugging Face")
            
            # Set the token in the environment for other libraries
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            
            from llama_cookbook import main
            # Call the main function with the train_config parameters
            # Add max_samples parameter to limit dataset size in llama_cookbook
            config_dict = train_config.__dict__.copy()
            # Set quantization to 8bit to match the INT8 base model
            config_dict['quantization'] = "8bit"
            logger.info("Setting quantization to 8bit to match the INT8 base model")
            
            if train_config.max_samples is not None:
                logger.info(f"Passing max_samples={train_config.max_samples} to llama_cookbook")
                # Add max_samples to the config dictionary
                config_dict['max_train_samples'] = train_config.max_samples
                config_dict['max_eval_samples'] = max(100, int(train_config.max_samples * 0.1))  # 10% for evaluation
            
            main(**config_dict)
        except Exception as e:
            logger.error(f"Error calling llama_cookbook.main: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return
        
        # Check if the script is proceeding to the next section
        # Add a flag to track execution flow
        execution_tracking = {}
        execution_tracking['checked_train_json'] = True
    else:
        # Load and preprocess the dataset
        logger.info(f"Loading dataset from {train_config.dataset_path}")
        
        try:
            # Directly load the dataset from the specified file
            logger.info(f"Attempting to load dataset for phase {train_config.phase}...")
            if train_config.phase == 1:
                logger.info(f"Loading phase 1 dataset from {train_config.phase1_file}")
                raw_dataset = load_dataset('text', data_files=train_config.phase1_file)
                logger.info(f"Successfully loaded phase 1 dataset: {raw_dataset}")
            else:
                logger.info(f"Loading phase 2 dataset from {train_config.phase2_file}")
                raw_dataset = load_dataset('text', data_files=train_config.phase2_file)
                logger.info(f"Successfully loaded phase 2 dataset: {raw_dataset}")
                
            # Limit the dataset size if max_samples is specified
            if train_config.max_samples is not None:
                logger.info(f"Limiting dataset to {train_config.max_samples} samples as specified")
                # Filter the dataset to the specified number of samples
                for split in raw_dataset.keys():
                    if len(raw_dataset[split]) > train_config.max_samples:
                        raw_dataset[split] = raw_dataset[split].select(range(train_config.max_samples))
                logger.info(f"Dataset limited to {train_config.max_samples} samples (approximately {train_config.max_samples * 5} words on average)")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Load tokenizer
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {train_config.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name or train_config.model_name,
            padding_side="right",
            use_fast=True,
        )
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Preprocess the dataset with proper padding and truncation
        preprocessed_dataset = preprocess_dataset(
            raw_dataset, 
            tokenizer, 
            max_length=train_config.context_length
        )
        
        # Save the preprocessed dataset to disk
        preprocessed_dataset.save_to_disk(preprocessed_path)
        
        # Update the dataset path to use the preprocessed dataset
        train_config.dataset_path = preprocessed_path
        logger.info(f"Saved preprocessed dataset to {preprocessed_path}")
        
        # Update the dataset path to point to the current directory
        original_dataset_path = train_config.dataset_path
        train_config.dataset_path = os.path.dirname(train_config.dataset_path)
        logger.info(f"Updated dataset_path from {original_dataset_path} to {train_config.dataset_path}")
        
        # Check if the dataset directory exists and has the required files
        logger.info(f"Checking if dataset directory exists at {train_config.dataset_path}")
        if os.path.exists(train_config.dataset_path):
            logger.info(f"Dataset directory exists. Contents: {os.listdir(train_config.dataset_path)}")
        else:
            logger.error(f"Dataset directory does not exist at {train_config.dataset_path}")
            
        # Check if we can access the llama_main function
        logger.info("Checking if llama_main function is accessible...")
        try:
            logger.info(f"llama_main function: {llama_main}")
            logger.info("llama_main function is accessible")
            
            # Check if the llama_main function has the expected signature
            import inspect
            sig = inspect.signature(llama_main)
            logger.info(f"llama_main function signature: {sig}")
            logger.info(f"llama_main function parameters: {list(sig.parameters.keys())}")
        except Exception as e:
            logger.error(f"Error accessing llama_main function: {str(e)}")
            
        # Check if the local llama_cookbook.py file exists
        cookbook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llama_cookbook.py')
        logger.info(f"Checking if local llama_cookbook.py exists at {cookbook_path}")
        if os.path.exists(cookbook_path):
            logger.info(f"Local llama_cookbook.py exists. Size: {os.path.getsize(cookbook_path)} bytes")
        else:
            logger.error(f"Local llama_cookbook.py does not exist at {cookbook_path}")

        # Check if phase1_data.txt or phase2_data.txt exists in the current directory
        phase1_file = train_config.phase1_file
        phase2_file = train_config.phase2_file
        logger.info(f"Phase 1 file path: {phase1_file}")
        logger.info(f"Phase 2 file path: {phase2_file}")
        
        # Track execution flow
        execution_tracking['checked_phase_files'] = True
        
        # We already have the JSONL files, so we can proceed with the fine-tuning
        logger.info("Using existing JSONL files for fine-tuning...")
        
        # We already have the JSONL files, so we can proceed with the fine-tuning
        logger.info("Skipping phase file processing and using existing JSONL files...")
        
        # The cookbook expects train.jsonl, val.jsonl, and test.jsonl in the formatted_data directory
        train_json_file = os.path.join(formatted_data_dir, "train.jsonl")
        val_json_file = os.path.join(formatted_data_dir, "val.jsonl")
        test_json_file = os.path.join(formatted_data_dir, "test.jsonl")
        
        # Verify that all required files exist
        logger.info(f"Verifying JSONL files for fine-tuning:\n  - train.jsonl exists: {os.path.exists(train_json_file)}\n  - val.jsonl exists: {os.path.exists(val_json_file)}\n  - test.jsonl exists: {os.path.exists(test_json_file)}")
        
        # Skip phase file processing and use existing JSONL files
        if os.path.exists(train_json_file) and os.path.exists(val_json_file) and os.path.exists(test_json_file):
            logger.info("All required JSONL files exist. Skipping phase file processing...")
            try:
                # Count the number of entries in each file
                with open(train_json_file, 'r', encoding='utf-8') as f:
                    train_count = sum(1 for _ in f)
                with open(val_json_file, 'r', encoding='utf-8') as f:
                    val_count = sum(1 for _ in f)
                with open(test_json_file, 'r', encoding='utf-8') as f:
                    test_count = sum(1 for _ in f)
                
                logger.info(f"Found {train_count} entries in train.jsonl, {val_count} in val.jsonl, and {test_count} in test.jsonl")
                
                # Set the dataset path to the formatted_data directory
                train_config.dataset_path = formatted_data_dir
                logger.info(f"Set dataset_path to {train_config.dataset_path}")
                
                # Proceed to fine-tuning
                logger.info("Proceeding to fine-tuning with existing JSONL files...")
                
                # Skip the rest of the phase file processing
                # Jump to the fine-tuning preparation section
                logger.info("========== PROCEEDING TO FINE-TUNING PREPARATION ==========")
                
                # Call the main function from llama_cookbook
                logger.info(f"Current train_config: {train_config.__dict__}")
                logger.info(f"Current FSDP config: {fsdp_config.__dict__}")
                logger.info(f"Current quant config: {quant_config.__dict__}")
                logger.info(f"Current wandb config: {wandb_config}")
                
                # Check CUDA availability
                if torch.cuda.is_available():
                    logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
                    logger.info(f"Current device: {torch.cuda.current_device()}")
                    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
                    logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
                    logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
                else:
                    logger.warning("CUDA is not available. Training will be slow.")
                
                # Call the main function
                from llama_cookbook import main
                main(**train_config.__dict__)
                
                return
            except Exception as e:
                logger.error(f"Error using existing JSONL files: {str(e)}")
                logger.info("Falling back to phase file processing...")
        
        try:
            
            # Process the text file based on the phase
            if train_config.phase == 1:
                # Phase 1: Each entry should be a translated paragraph followed by English paragraph
                # Format: {"text": "<translated_text>\n\n<english_text>"}
                data = []
                logger.info("Processing Phase 1 data: translated paragraph followed by English paragraph")
                
                # Try to detect the format of the file
                # If each paragraph is on a single line and paragraphs are separated by blank lines
                if len([line for line in lines if line.strip()]) / 2 < len(lines) * 0.8:
                    # Format with blank lines between paragraphs
                    logger.info("Detected format: paragraphs separated by blank lines")
                    current_paragraph = []
                    for line in lines:
                        if line.strip():
                            current_paragraph.append(line.strip())
                        elif current_paragraph:  # Empty line and we have content
                            if len(current_paragraph) >= 2:
                                # Assuming first half is Yanomami, second half is English
                                mid = len(current_paragraph) // 2
                                yanomami_text = " ".join(current_paragraph[:mid])
                                english_text = " ".join(current_paragraph[mid:])
                                data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                            current_paragraph = []
                    
                    # Don't forget the last paragraph if there's no final blank line
                    if current_paragraph and len(current_paragraph) >= 2:
                        mid = len(current_paragraph) // 2
                        yanomami_text = " ".join(current_paragraph[:mid])
                        english_text = " ".join(current_paragraph[mid:])
                        data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                else:
                    # Assuming alternating lines: translated text followed by English text
                    logger.info("Detected format: alternating lines (Yanomami, English, Yanomami, ...)")
                    i = 0
                    while i < len(lines):
                        if i + 1 < len(lines):
                            yanomami_text = lines[i].strip()
                            english_text = lines[i+1].strip()
                            if yanomami_text and english_text:  # Skip empty lines
                                data.append({"text": f"{yanomami_text}\n\n{english_text}"})
                            i += 2
                        else:
                            i += 1
            else:
                # Phase 2: Alternating sentences in English and Yanomami
                # Format: {"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}
                logger.info("Processing Phase 2 data: alternating sentences in English and Yanomami")
                data = []
                
                # Try to detect if the file already has alternating sentences on each line
                # or if we need to process it differently
                for line in lines:
                    if line.strip():
                        # Just use each non-empty line as is
                        data.append({"text": line.strip()})
                
                logger.info(f"Processed {len(data)} lines for Phase 2 training")
            
            # Split the data into train, validation, and test sets (80/10/10 split)
            # This follows the Meta Llama cookbook approach for extending to new languages
            import random
            random.seed(train_config.seed)  # For reproducibility
            random.shuffle(data)
            
            total_samples = len(data)
            train_size = int(0.8 * total_samples)
            val_size = int(0.1 * total_samples)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            logger.info(f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test samples")
            
            # Write the JSON files in JSONL format as required by the cookbook
            def write_jsonl(data_list, file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data_list:
                        f.write(json.dumps(item) + '\n')
                logger.info(f"Created {file_path} with {len(data_list)} entries")
            
            write_jsonl(train_data, train_json_file)
            write_jsonl(val_data, val_json_file)
            write_jsonl(test_data, test_json_file)
            
            # Update the dataset path to point to the formatted_data directory
            original_dataset_path = train_config.dataset_path
            train_config.dataset_path = formatted_data_dir
            
            # Update the output directory to be phase-specific
            original_output_dir = train_config.output_dir
            train_config.output_dir = os.path.join(original_output_dir, f"phase{train_config.phase}")
            os.makedirs(train_config.output_dir, exist_ok=True)
            
            logger.info(f"Updated dataset path from {original_dataset_path} to: {train_config.dataset_path}")
            logger.info(f"Updated output directory from {original_output_dir} to: {train_config.output_dir}")
            
        except Exception as e:
            logger.error(f"Error processing the data file: {str(e)}")
            sys.exit(1)
        
        # Track execution flow before setting up training parameters
        logger.info("========== SETTING UP TRAINING PARAMETERS ==========")
        logger.info(f"Execution tracking so far: {execution_tracking}")
        execution_tracking['starting_parameter_setup'] = True
        
        # According to the cookbook, we need to use specific parameters for training
        # Based on the OpenHathi training parameters mentioned in the cookbook
        # and the parameters used in the llama-cookbook finetuning.py file
        
        # Start with the parameters specified in the cookbook
        kwargs = {
            # Model and tokenizer parameters
            "model_name": train_config.model_name,
            "tokenizer_name": train_config.tokenizer_name,
            "output_dir": train_config.output_dir,
            
            # Dataset parameters
            "dataset_path": train_config.dataset_path,  # Path to the formatted_data directory
            "dataset_format": "jsonl",  # Format used in the cookbook
            "dataset_text_field": "text",  # Field name in the JSON file
            "batching_strategy": "packing",  # From cookbook finetuning.py
            "context_length": 4096,  # From cookbook
            
            # Training parameters
            "lr": 2e-4,  # maximum learning rate from cookbook
            "weight_decay": 0.1,  # from cookbook
            "gamma": 0.85,  # LR decay factor
            "num_train_epochs": train_config.num_train_epochs,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "per_device_train_batch_size": train_config.per_device_train_batch_size,
            "per_device_eval_batch_size": train_config.per_device_eval_batch_size,
            "warmup_steps": 50,  # Number of warmup steps for learning rate scheduler
            "run_validation": True,  # Run validation during training
            "seed": train_config.seed,
            
            # PEFT (LoRA) parameters
            "use_peft": True,  # We need PEFT for LoRA
            "peft_method": "lora",  # Using LoRA as specified in cookbook
            "lora_r": 128,  # from cookbook
            "lora_alpha": 64,  # from cookbook
            "lora_dropout": 0.05,  # from cookbook
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],  # from cookbook
            
            # FSDP parameters (for distributed training)
            "enable_fsdp": False,  # Disable FSDP for single-GPU training
            "low_cpu_fsdp": False,
            "use_fast_kernels": True,  # Enable fast kernels for training
            "use_fp16": False,  # Use BF16 instead as recommended in cookbook
        }
        
        # Log the kwargs being passed to main
        logger.info(f"Passing the following parameters to llama_cookbook.finetuning.main: {kwargs.keys()}")
        
        # Remove parameters that might not be recognized
        # Only keep the necessary parameters for dataset processing
        for param in ["min_lr", "beta1", "beta2", "block_size", "dtype", "train"]:
            if param in kwargs:
                del kwargs[param]
        
        # Set environment variable for PyTorch memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # Ensure PEFT is enabled to reduce memory usage
        kwargs['use_peft'] = True
        
        # Enable mixed precision for memory efficiency
        kwargs['mixed_precision'] = True
        
        # Set quantization to 8bit to match the INT8 base model
        kwargs['quantization'] = "8bit"
        logger.info("Setting quantization to 8bit to match the INT8 base model")
        
        # Run the fine-tuning process
        try:
            # First try with all parameters
            logger.info("========== STARTING FINE-TUNING PROCESS ==========")
            logger.info("Attempting to run with all parameters from the cookbook...")
            
            # Log all the kwargs being passed to llama_main
            logger.info("Parameters being passed to llama_main:")
            for key, value in kwargs.items():
                logger.info(f"  {key}: {value}")
            
            # Check if dataset exists at the path
            logger.info(f"Checking if dataset exists at {kwargs.get('dataset_path', 'Not specified')}")
            if 'dataset_path' in kwargs and os.path.exists(kwargs['dataset_path']):
                logger.info(f"Dataset path exists. Contents: {os.listdir(kwargs['dataset_path'])}")
                
                # Check specifically for the jsonl files required by the cookbook
                train_jsonl = os.path.join(kwargs['dataset_path'], 'train.jsonl')
                val_jsonl = os.path.join(kwargs['dataset_path'], 'val.jsonl')
                test_jsonl = os.path.join(kwargs['dataset_path'], 'test.jsonl')
                
                logger.info(f"Checking for required JSONL files:\n  - train.jsonl exists: {os.path.exists(train_jsonl)}\n  - val.jsonl exists: {os.path.exists(val_jsonl)}\n  - test.jsonl exists: {os.path.exists(test_jsonl)}")
                
                # If we have train.json instead of train.jsonl, we need to fix it
                train_json = os.path.join(kwargs['dataset_path'], 'train.json')
                if os.path.exists(train_json) and not os.path.exists(train_jsonl):
                    logger.warning(f"Found train.json but not train.jsonl. Converting format...")
                    try:
                        # Read train.json and write to train.jsonl
                        with open(train_json, 'r', encoding='utf-8') as f_in:
                            data = [json.loads(line) for line in f_in]
                        
                        # Create val.jsonl and test.jsonl from the same data
                        import random
                        random.seed(train_config.seed)
                        random.shuffle(data)
                        
                        total = len(data)
                        train_size = int(0.8 * total)
                        val_size = int(0.1 * total)
                        
                        train_data = data[:train_size]
                        val_data = data[train_size:train_size+val_size]
                        test_data = data[train_size+val_size:]
                        
                        # Write the files
                        with open(train_jsonl, 'w', encoding='utf-8') as f:
                            for item in train_data:
                                f.write(json.dumps(item) + '\n')
                        
                        with open(val_jsonl, 'w', encoding='utf-8') as f:
                            for item in val_data:
                                f.write(json.dumps(item) + '\n')
                                
                        with open(test_jsonl, 'w', encoding='utf-8') as f:
                            for item in test_data:
                                f.write(json.dumps(item) + '\n')
                                
                        logger.info(f"Successfully converted train.json to train.jsonl, val.jsonl, and test.jsonl")
                    except Exception as e:
                        logger.error(f"Error converting train.json to jsonl format: {str(e)}")
            elif 'dataset_path' in kwargs:
                logger.error(f"Dataset path does not exist: {kwargs['dataset_path']}")
            
            logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            logger.info("Calling llama_main function now...")
            result = llama_main(**kwargs)
            logger.info("Successfully completed llama_main function call!")
        except (TypeError, RuntimeError) as e:
            logger.error(f"Error calling llama_main: {str(e)}")
            
            if isinstance(e, RuntimeError) and 'CUDA out of memory' in str(e):
                logger.error("CUDA out of memory error detected. Reducing context length and batch size...")
                
                # Reduce context length to save memory
                kwargs['context_length'] = 2048  # Reduce from 4096 to 2048
                logger.info(f"Reduced context_length to {kwargs['context_length']}")
                
                # Increase gradient accumulation to compensate for smaller batch size
                kwargs['gradient_accumulation_steps'] = 4
                logger.info(f"Increased gradient_accumulation_steps to {kwargs['gradient_accumulation_steps']}")
                
                # Clear cache again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache again")
                
                try:
                    logger.info(f"Retrying with reduced memory parameters: {kwargs.keys()}")
                    result = llama_main(**kwargs)
                except Exception as e2:
                    logger.error(f"Still encountering errors: {str(e2)}")
                    logger.error("Trying with minimal parameters...")
                    
                    # Try with minimal parameters focused on memory efficiency
                    minimal_kwargs = {
                        "model_name": train_config.model_name,
                        "tokenizer_name": train_config.tokenizer_name,
                        "output_dir": train_config.output_dir,
                        "dataset_path": train_config.dataset_path,
                        "use_peft": True,
                        "peft_method": "lora",
                        "lora_r": 64,  # Reduced from 128
                        "lora_alpha": 32,  # Reduced from 64
                        "context_length": 1024,  # Further reduced
                        "per_device_train_batch_size": 1,
                        "gradient_accumulation_steps": 8,
                        "mixed_precision": True,
                        "quantization": "8bit",  # Set to 8bit to match the INT8 base model
                    }
                    logger.info(f"Trying with minimal parameters: {minimal_kwargs.keys()}")
                    result = llama_main(**minimal_kwargs)
            else:
                # Handle TypeError (parameter mismatch)
                logger.error("The function signature may have changed. Trying with a subset of parameters...")
                
                # Remove parameters that might not be recognized
                for param in ["min_lr", "beta1", "beta2", "block_size", "dtype"]:
                    if param in kwargs:
                        del kwargs[param]
                
                logger.info(f"Trying with reduced parameters: {kwargs.keys()}")
                result = llama_main(**kwargs)
        
        logger.info(f"Fine-tuning completed with result: {result}")
        
        return result

if __name__ == "__main__":
    try:
        main_cli()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
