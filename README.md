# Spanish-to-Arabic Neural Machine Translation

![BLEU Score](https://img.shields.io/badge/BLEU-27.06-brightgreen)
![Model](https://img.shields.io/badge/Model-MarianMT-blue)
![Languages](https://img.shields.io/badge/Languages-Spanish--Arabic-orange)

A high-quality neural machine translation system for translating Spanish text to Arabic, built using the MarianMT transformer architecture and optimized with advanced beam search techniques.

## Overview

This project implements a Spanish→Arabic translation system by fine-tuning the MarianMT model on a parallel corpus combining UN documents and OpenSubtitles data. The system achieves a BLEU score of 27.06 on the test set, performing particularly well on formal and institutional text.

### Features
- Fine-tuned MarianMT model specifically for Spanish-to-Arabic translation
- Optimized beam search with width 5 for improved translation quality
- Easy-to-use translation scripts and evaluation tools
- Detailed analysis of model performance across different text types
- Comprehensive notebooks documenting the entire process

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/spanish-arabic-translation.git
cd spanish-arabic-translation

# Create a virtual environment (optional but recommended)
conda create -n spanish_arabic python=3.9
conda activate spanish_arabic

# Install dependencies
pip install -r requirements.txt

```
## Quick Start
Translate Spanish text to Arabic:

```python
from scripts.translate import translate_spanish_to_arabic

text = "La paz y la cooperación internacional son fundamentales para el desarrollo sostenible."
translation = translate_spanish_to_arabic(text)
print(translation)
# Output: السلام والتعاون الدولي أمران أساسيان للتنمية المستدامة.

```

Or use the command line interface:

```bash
python scripts/translate.py --text "¿Cómo estás hoy?"

```

## Project Structure

- notebooks/: Jupyter notebooks with the full development process

  - 01_data_preprocessing.ipynb: Data cleaning, normalization, and preparation
  - 02_model_implementation.ipynb: Model training, evaluation, and optimization


- scripts/: Utility scripts for using the model
- models/: Directory for model checkpoints (download separately)
- data/: Data files used for training and evaluation
- results/: Evaluation results and visualizations

## Data
The model was trained on approximately 17,000 Spanish-Arabic parallel sentences from:

UN Parallel Corpus (UN v1.0 ar-es)

OpenSubtitles corpus

## Model Training
Fine-tuned the MarianMT Helsinki-NLP/opus-mt-es-ar model with the following parameters:

Learning rate: 5e-5

Batch size: 16

Training epochs: 4

Optimizer: AdamW with weight decay 0.01

Mixed precision training (FP16)

## Acknowledgments
Helsinki-NLP for the MarianMT pre-trained models

Hugging Face for the Transformers library

UN Parallel Corpus and OpenSubtitles for the training data
