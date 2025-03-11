#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for comparing base and fine-tuned models for Yanomami language.
This script supports testing both base models and PEFT fine-tuned models.
"""

import os
import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Test models for Yanomami language")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "finetuned"],
        default="base",
        help="Type of model to test (base or finetuned)"
    )
    parser.add_argument(
        "--base_model_name", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Name of the base model"
    )
    parser.add_argument(
        "--finetuned_model_path", 
        type=str, 
        default="./output/phase1",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training phase (1 or 2)"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face token for accessing gated models"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Use CPU only (slower but more reliable)"
    )
    return parser.parse_args()

def load_model(args):
    # Set device
    device = "cpu" if args.cpu_only else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.model_type == "base":
        # Load tokenizer directly from the base model
        print(f"Loading tokenizer from {args.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, token=args.hf_token)
        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
        
        # Load model
        print(f"Loading base model from {args.base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            token=args.hf_token
        )
    else:  # finetuned model
        # First, check if this is a PEFT model
        is_peft_model = os.path.exists(os.path.join(args.finetuned_model_path, "adapter_config.json"))
        print(f"Is PEFT model: {is_peft_model}")
        
        if is_peft_model:
            # Load tokenizer from the base model
            print(f"Loading tokenizer from {args.base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, token=args.hf_token)
            print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
            
            # Load base model
            print(f"Loading base model from {args.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=args.hf_token
            )
            
            # Determine target vocab size from adapter config
            adapter_config_path = os.path.join(args.finetuned_model_path, "adapter_config.json")
            target_vocab_size = 134814  # Default based on error message
            
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    if 'vocab_size' in adapter_config:
                        target_vocab_size = adapter_config['vocab_size']
                        print(f"Found vocab_size in adapter_config: {target_vocab_size}")
                except Exception as e:
                    print(f"Error reading adapter_config.json: {e}")
            
            # Resize the base model to match the target size
            if base_model.get_input_embeddings().weight.shape[0] != target_vocab_size:
                print(f"Resizing model embeddings from {base_model.get_input_embeddings().weight.shape[0]} to {target_vocab_size}")
                base_model.resize_token_embeddings(target_vocab_size)
            
            # Load PEFT model
            try:
                print(f"Loading PEFT model from {args.finetuned_model_path}")
                model = PeftModel.from_pretrained(base_model, args.finetuned_model_path)
            except Exception as e:
                print(f"Error loading PEFT model: {e}")
                print("Trying alternative loading method...")
                try:
                    peft_config = PeftConfig.from_pretrained(args.finetuned_model_path)
                    model = PeftModel.from_pretrained(base_model, args.finetuned_model_path, config=peft_config)
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    return None, None, device
        else:
            # Load regular model
            print(f"Loading regular model from {args.finetuned_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path, token=args.hf_token)
            print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.finetuned_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=args.hf_token
            )
    
    if device == "cpu" and (args.model_type == "base" or not is_peft_model):
        model = model.to(device)
    
    return model, tokenizer, device

def main():
    args = parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args)
    
    if model is None or tokenizer is None:
        print("Model or tokenizer failed to load. Exiting.")
        return
    
    # Different prompting strategies based on model type and phase
    test_phrases = [
        "Hello, how are you?",
        "My name is John",
        "I want to learn Yanomami",
        "The forest is beautiful",
        "Thank you for your help"
    ]
    
    for phrase in test_phrases:
        try:
            print(f"\n=== Testing with phrase: '{phrase}' ===")
            
            # Different prompting strategies based on model type and phase
            if args.model_type == "base":
                prompt = f"Translate the following English text to Yanomami: {phrase}\nYanomami translation:"
            elif args.model_type == "finetuned" and args.phase == 1:
                # Phase 1 format: expect Yanomami text before English
                prompt = f"\n\n{phrase}"
            elif args.model_type == "finetuned" and args.phase == 2:
                # Phase 2 format: alternating sentences
                prompt = f"{phrase} "
            
            print(f"Using prompt: '{prompt}'")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated text: '{generated_text}'")
    
            # Extract translation based on model type and phase
            if args.model_type == "base":
                translation = generated_text.split("Yanomami translation:")[-1].strip()
            elif args.model_type == "finetuned" and args.phase == 1:
                # For Phase 1, the translation should be before the prompt
                translation = generated_text[:generated_text.find(prompt)].strip()
            elif args.model_type == "finetuned" and args.phase == 2:
                # For Phase 2, the translation should be after the prompt
                translation = generated_text[len(prompt):].strip()
            
            print(f"Extracted translation: '{translation}'")
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
