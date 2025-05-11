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
