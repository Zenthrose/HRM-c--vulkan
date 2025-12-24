# HRM Data Directory

This directory contains training data, corpora, and datasets for the HRM (Hierarchical Reasoning Model) system.

## Directory Structure

```
data/
├── text/                 # Text corpora for language training
│   ├── raw/             # Raw, unprocessed text data
│   ├── processed/       # Processed and cleaned text data
│   ├── training/        # Training dataset splits
│   ├── validation/      # Validation dataset splits
│   └── test/            # Test dataset splits
├── models/              # Saved model checkpoints
├── logs/                # Training logs and metrics
└── temp/                # Temporary processing files
```

## Text Corpora

The HRM system uses diverse text corpora to develop reasoning, conversation, and knowledge capabilities.

### Included Corpora

1. **Literary Works** - Classic literature for narrative understanding
2. **Programming Code** - Source code for algorithmic reasoning
3. **Conversations** - Educational dialogues for conversational AI
4. **Reasoning Exercises** - Logic and critical thinking examples
5. **Scientific Content** - Research and methodology texts

### Data Preparation

Use the provided scripts to prepare training data:

```bash
# Prepare language dataset with default settings
./prepare_language_dataset.sh

# Prepare with custom data directory
./prepare_language_dataset.sh --data-dir /path/to/custom/data

# Setup character-level training environment
./setup_character_training.sh

# Setup with automatic Python dependency installation
./setup_character_training.sh --install-deps
```

### Corpus Download Scripts

For large external corpora, use these commands:

```bash
# Download OpenWebText2 sample (diverse web content)
curl -L "https://huggingface.co/datasets/openwebtext/resolve/main/plain_text/train-00000-of-00001.parquet" \
     -o data/text/raw/openwebtext_sample.parquet

# Download BookCorpus (literary content)
# Note: Requires HuggingFace account and API token
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bookcorpus/bookcorpus',
    repo_type='dataset',
    local_dir='data/text/raw/books'
)
"

# Generate synthetic conversations
python scripts/generate_conversation_data.py --output data/text/raw/conversations/generated_conversations.txt
```

## Model Storage

Trained models are stored in the `models/` directory:

```
models/
├── character_model_epoch_1.ckpt
├── character_model_epoch_2.ckpt
└── hrm_model_v1.bin
```

Models are automatically saved during training with timestamps and version numbers.

## Log Files

Training logs and metrics are stored in `logs/`:

```
logs/
├── character_training_20241207.log
├── hrm_training_metrics.json
└── system_performance.log
```

## Data Expansion

To expand the training dataset:

1. **Add new text sources** to `data/text/raw/`
2. **Run preprocessing** with `prepare_language_dataset.sh`
3. **Validate data quality** using the evaluation tools
4. **Retrain models** with expanded datasets

## Quality Guidelines

- **Diversity**: Include multiple domains (technical, literary, conversational)
- **Cleanliness**: Remove formatting artifacts, ensure UTF-8 encoding
- **Balance**: Maintain reasonable proportions across different content types
- **Size**: Start with smaller datasets for testing, scale up for production

## Troubleshooting

**Common Issues:**

1. **Permission errors**: Ensure write access to data directory
2. **Memory issues**: Large corpora may require system with sufficient RAM
3. **Encoding problems**: Ensure all text files are UTF-8 encoded
4. **Path issues**: Use absolute paths for custom data directories

**Validation Commands:**

```bash
# Check data statistics
wc -l data/text/processed/training_corpus.txt
ls -la data/text/raw/

# Validate UTF-8 encoding
file data/text/processed/*.txt

# Check for common issues
grep -r "�" data/text/  # Look for encoding errors
```

## Contributing

When adding new data sources:

1. Follow the directory structure
2. Update this README with new corpus descriptions
3. Include download scripts or instructions
4. Test preprocessing pipeline
5. Validate with evaluation tools