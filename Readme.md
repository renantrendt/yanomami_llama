# Yanomami Language Extension for Llama 3.1

## Test the model

https://yanomami.bernardoserrano.com

## Project Overview

This project aims to extend the Meta Llama 3.1 model to support the Yanomami language, an indigenous language spoken by approximately 35,000 people in the Amazon rainforest regions of Brazil and Venezuela. By fine-tuning Llama 3.1 on Yanomami language data, we create a multilingual model capable of understanding and generating text in both Yanomami and English.

1. We trained a model Llama 3.1 8B INT8 using 8xA100 GPUs on Lambdalabs
2. We created a chat interface for the model using assistant-ui https://github.com/renantrendt/yanomami-chat
3. We are hosting the model on Lambdalabs for inference
4. We are adding Qdrant to the model for knowledge retrieval
5. We plan to create an app that runs offline because on the forest there is no internet connection

We tried to fine tune GPT2-small but we failed. Now we hope that this model will.

The project follows the Meta Llama cookbook approach for extending language models to new languages, implementing a two-phase training process:

1. **Phase 1**: Learning to translate paragraphs (translated text as context, generate original text)
2. **Phase 2**: Bilingual next token prediction (alternating sentences in both languages)

### Phase 1: Translation Learning

**Objective**: Teach the model to understand the relationship between the new language (Yanomami) and English.

**Data Format**: `{"text": "<translated_text>\n\n<english_text>"}`
- The model is given translated text as context and learns to generate the original English text.
- Example: Yanomami text followed by two newlines, then the corresponding English text.

**Learning Focus**: 
- Basic vocabulary and grammar of the new language
- Mapping between concepts in both languages
- Understanding the structure of the new language

**File Used**: `formatted_data/phase1_data.txt`

### Phase 2: Bilingual Next Token Prediction

**Objective**: Improve the model's ability to seamlessly switch between languages and generate coherent text in both.

**Data Format**: `{"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}`
- Alternating sentences in both languages
- The model learns to predict the next token regardless of which language it's in

**Learning Focus**:
- Code-switching (moving between languages)
- Maintaining context across language boundaries
- Generating coherent text in both languages

**File Used**: `formatted_data/phase2_data.txt`

This two-phase approach is designed to gradually build the model's capabilities in the new language, first establishing basic understanding and translation abilities, then developing more sophisticated bilingual capabilities.

## Installation Instructions

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (training was performed on 8xA100 GPUs)
- Hugging Face account with access to Llama 3.1 models

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llama-yanomami-extension.git
   cd llama-yanomami-extension
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   sudo apt-get update
   sudo apt-get install -y build-essential python3-dev
   ```

3. Set up environment variables:
   ```bash
   export HF_DATASETS_TRUST_REMOTE_CODE=True
   ```

4. Move necessary files to the correct locations:
   ```bash
   # If using llama_cookbook
   mv extend_language/samsum_dataset.py ~/.local/lib/python3.10/site-packages/llama_cookbook/datasets/
   ```

## Usage Guide

### Data Preparation

#### Initial Data Processing

The data preparation process involves several steps to transform raw Yanomami-English data into the formats required for the two-phase training approach:

1. **Raw Data Extraction**: The initial scripts extract Yanomami words/phrases and their English translations from structured data with specific tags:
   - `<WORD>...</WORD>`: Contains Yanomami words or phrases
   - `<DEFINITION>...</DEFINITION>`: Contains English translations
   - `<EXAMPLE_YANOMAMI>...</EXAMPLE_YANOMAMI>` and `<EXAMPLE_TRANSLATION>...</EXAMPLE_TRANSLATION>`: Contain example sentences

2. **Phase-Specific Formatting**:
   - **Phase 1**: Creates data in the format `{Yanomami word/phrase} = {English translation}` and saves to `phase1_data.txt`
   - **Phase 2**: Creates bilingual text with alternating sentences between English and Yanomami and saves to `phase2_data.txt`

3. **Data Processing Commands**:
   ```bash
   # Extract and process raw text for tokenizer training
   python prepare_data.py --dataset_path ./dataset/validation.jsonl --save_path ./data
   
   # Create formatted data for Phase 1
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 1
   
   # Create formatted data for Phase 2
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 2
   ```

This preparation process ensures the data is properly formatted according to the Meta Llama cookbook approach for extending language models to new languages.

#### Treatment of Diacritical Marks

The Yanomami language contains various diacritical marks that require special handling during tokenization and processing:

1. **Unicode Normalization**: The tokenizer applies the `nmt_nfkc_cf` normalization rule (Normalization Form KC with Case Folding) which:
   - Ensures consistent representation of characters with diacritical marks
   - Decomposes and then recomposes characters in a canonical form
   - Helps maintain consistency across different text sources

2. **Character Coverage**: The tokenizer is trained with a high character coverage value (default: 1.0) to ensure all special characters and diacritical marks in Yanomami are properly recognized.

3. **UTF-8 Encoding**: All text files are processed using UTF-8 encoding to properly handle the full range of Unicode characters present in Yanomami text.

4. **NFC Encoding**: When testing the tokenizer, text is read using NFC (Normalization Form C) encoding to ensure proper handling of combining diacritical marks.

This careful handling of diacritical marks is essential for properly representing the Yanomami language, which contains unique phonetic features that must be preserved during the tokenization and training process.

#### Requirement to run llama-cookbook
   
2. Set up SamSum dataset for testing and benchmarking:
   ```bash
   # Clone the SamSum dataset repository
   git clone https://huggingface.co/datasets/Samsung/samsum
   
   # Set environment variable for remote code execution
   export HF_DATASETS_TRUST_REMOTE_CODE=True
   
   # Process the SamSum dataset
   python extend_language/samsum/samsum.py
   
   # Run the SamSum dataset preparation script
   python /home/ubuntu/.local/lib/python3.10/site-packages/llama_cookbook/datasets/samsum_dataset.py
   ```
### Train the tokenizer
3. Train a tokenizer for Yanomami:
   ```bash
   python train_tokenizer.py --input_file ./data/yan.txt --save_path ./yanomami_tokenizer --vocab_size 8000
   ```

4. Extend the base Llama tokenizer with Yanomami tokens:
   ```bash
   python extend_tokenizer_v2.py \ 
       --base_model_name meta-llama/Meta-Llama-3.1-8B \ 
       --new_tokenizer_path ./yanomami_tokenizer \ 
       --extended_tokenizer_save_path ./extended_tokenizer \ 
       --hf_token YOUR_HF_TOKEN
   ```

5. Fix the tokenizer (to address 'OrderedVocab contains holes' warning):
   ```bash
   python fix_tokenizer.py --tokenizer_path ./extended_tokenizer --output_path ./fixed_tokenizer
   ```

6. Prepare training data for both phases:
   ```bash
   # Phase 1: Translation format
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 1
   
   # Phase 2: Bilingual next token prediction format
   python prepare_training_data.py --input_file ./dataset/train.jsonl --output_dir ./formatted_data --phase 2
   ```

### Training

#### Phase 1: Translation Learning

```bash
python yanomami_finetune.py --phase=1 --use_peft
```

#### Phase 2: Bilingual Next Token Prediction

Based on the learning of phase 1 now it will predict the next token

```bash
python yanomami_finetune.py --phase=2 --use_peft
```

## Training Results




### Model Saving

After training completes, the model is saved to the specified output directory (e.g., `./output/phase1` for Phase 1 training and so on). This directory contains:

- **adapter_config.json**: Configuration for the LoRA adapter
- **adapter_model.bin**: The trained LoRA weights
- **special_tokens_map.json**: Mapping of special tokens
- **tokenizer_config.json**: Tokenizer configuration
- **tokenizer.json**: The tokenizer data

These files can be loaded using the Hugging Face Transformers library for inference or further training.

## Model Implementation Details

This project implements the Meta Llama cookbook approach for extending language models to new languages. Key components include:

### Data Format

- **Phase 1**: `{"text": "<translated_text>\n\n<english_text>"}`
- **Phase 2**: `{"text": "<sentence1_en> <sentence1_yanomami> <sentence2_en> ..."}`

### Training Parameters

Following the cookbook recommendations:

- Learning rate: 2e-4
- LoRA rank: 128
- LoRA alpha: 64
- LoRA target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj
- Context length: 4096
- Training hardware: 8xA100 GPUs
- Mixed precision: BF16
- Quantization: 8-bit (QLoRA)
- DeepSpeed Zero-2 optimization

### Memory Optimization

The training process uses several memory optimization techniques:

- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 8-bit quantization (QLoRA)
- DeepSpeed Zero-2 for distributed training
- Gradient accumulation
- Mixed precision training

## Ethical Considerations and Limitations

### Ethical Considerations

1. **Cultural Preservation**: This project contributes to the digital preservation of the Yanomami language, supporting linguistic diversity and cultural heritage.

2. **Informed Consent**: Ensure that any Yanomami language data used has been collected with proper informed consent from native speakers and communities.

3. **Representation**: The model should accurately represent Yanomami language and culture without perpetuating stereotypes or misrepresentations.

4. **Access**: Consider how to make the resulting model accessible to Yanomami communities who could benefit from it.

### Limitations

1. **Data Scarcity**: Limited availability of high-quality Yanomami language data may affect model performance.

2. **Cultural Nuance**: The model may not capture all cultural nuances and contextual meanings specific to Yanomami culture.

3. **Dialect Variation**: The Yanomami language has several dialects, and the model may not represent all of them equally.

4. **Technical Requirements**: The computational resources required for inference may limit accessibility in remote areas where many Yanomami communities are located.

5. **Evaluation Challenges**: Limited availability of native Yanomami speakers for model evaluation may affect quality assessment.

## Acknowledgments

This project follows the approach outlined in the [Meta Llama Cookbook](https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/multilingual/README.md) for extending language models to new languages.

## Testing the Models

After training the model, you can evaluate its performance using the provided testing scripts. These scripts allow you to compare the base model with the fine-tuned models (Phase 1 and Phase 2).

### Using the test_models.py Script

The `test_models.py` script provides a simple way to test both the base model and fine-tuned models with different prompting strategies based on the training phase.

#### Testing the Base Model

```bash
python test_models.py --model_type base --base_model_name meta-llama/Meta-Llama-3.1-8B --hf_token YOUR_HF_TOKEN
```

#### Testing the Phase 1 Fine-tuned Model

```bash
python test_models.py --model_type finetuned --base_model_name meta-llama/Meta-Llama-3.1-8B --finetuned_model_path ./output/phase1 --phase 1 --hf_token YOUR_HF_TOKEN
```

#### Testing the Phase 2 Fine-tuned Model

```bash
python test_models.py --model_type finetuned --base_model_name meta-llama/Meta-Llama-3.1-8B --finetuned_model_path ./output/phase2 --phase 2 --hf_token YOUR_HF_TOKEN
```

If you encounter CUDA errors, add the `--cpu_only` flag to run the model on CPU instead:

```bash
python test_models.py --model_type finetuned --base_model_name meta-llama/Meta-Llama-3.1-8B --finetuned_model_path ./output/phase1 --phase 1 --hf_token YOUR_HF_TOKEN --cpu_only
```

### Using the evaluate_model.py Script

For a more comprehensive evaluation using BLEU scores, you can use the `evaluate_model.py` script:

```bash
python evaluate_model.py --base_model_name meta-llama/Meta-Llama-3.1-8B --finetuned_model_path ./output/phase1 --test_file ./dataset/test.jsonl --hf_token YOUR_HF_TOKEN
```

### Inspecting the Models

To inspect the tokenizer and model files without generating text (useful for debugging):

```bash
python inspect_model.py --tokenizer_path ./fixed_tokenizer --hf_token YOUR_HF_TOKEN
python inspect_model.py --model_type phase1 --model_path ./output/phase1 --tokenizer_path ./fixed_tokenizer --hf_token YOUR_HF_TOKEN
```

### Understanding the Results

When comparing the models, look for these indicators of improvement:

1. **Vocabulary Recognition**: The fine-tuned model should better recognize Yanomami words, especially those with diacritical marks.
2. **Translation Quality**: Phase 1 model should show improved translation capabilities compared to the base model.
3. **Code-Switching**: Phase 2 model should demonstrate better ability to switch between languages within the same context.
4. **Handling of Diacritical Marks**: All models should properly handle Yanomami's special characters and diacritical marks.

## License

This project is licensed under [LICENSE TYPE] - see the LICENSE file for details.
