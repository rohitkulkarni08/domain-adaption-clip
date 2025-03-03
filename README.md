# Domain Adaptation of CLIP for Multimodal Visual Tasks

## Overview

This project explores the **adaptation of CLIP (Contrastive Language-Image Pretraining)** for two vision-language tasks:
1. **Visual Question Answering (VQA)** - Generating responses to textual questions based on image content.
2. **Image Captioning** - Generating descriptive captions for images.

Domain adaptation techniques, including **adversarial training**, were applied to improve model performance on task-specific datasets.

## Features (https://drive.google.com/drive/folders/1yRuCdcxdqBHXK3icI9Chyi9H4D5VzKgM?usp=sharing)[Google Drive] for the datasets as files were hard to upload on GitHub

- **Visual Question Answering (VQA)**: Uses the **Visual Genome dataset** to generate image-based answers.
- **Image Captioning**: Trained on **Laion and Flickr30k datasets** for text-based caption generation.
- **Domain Adaptation**:
  - **Contrastive Loss**: Aligns image and text embeddings.
  - **Adversarial Training**: Reduces domain shift between pretraining and task-specific datasets.
- **Empirical Evaluation**: BLEU scores and accuracy metrics were used for model performance analysis.

## System Architecture

- **CLIP Model**:
  - **Image Encoder**: Vision Transformer (ViT) or ResNet for image embeddings.
  - **Text Encoder**: Transformer-based model for text embeddings.
- **Task-Specific Adaptation**:
  - **VQA Model**: Extends CLIP with a classifier for answer prediction.
  - **Captioning Model**: Uses a Transformer-based decoder for sequence generation.
- **Training Strategy**:
  - **Pretraining on Large Datasets (LAION, Visual Genome)**.
  - **Fine-tuning on Task-Specific Datasets (Flickr30k, Visual Genome)**.
  - **Domain Discriminator** for adversarial adaptation.

## Training & Evaluation

### VQA Model
- **Training Dataset**: 344,749 image-question-answer samples.
- **Optimization**: AdamW optimizer, batch size 32, learning rate 5×10⁻⁵.
- **Results**:
  - Achieved a training loss of **0.33**.
  - Outperformed **zero-shot CLIP** on VQA tasks.

### Image Captioning Model
- **Pretraining**: LAION dataset with Mean Squared Error (MSE) loss.
- **Fine-tuning**: Flickr30K dataset with adversarial domain adaptation.
- **Results**:
  - BLEU score improved from **0.2805** to **0.4116** with adversarial training.
  - Adversarial loss increased from **0.3024** to **0.9251**, showing effective domain alignment.

## Advantages & Challenges

### Advantages
- Domain adaptation significantly improves performance on task-specific datasets.  
- Adversarial training reduces domain shift, enhancing generalization.  
- Model achieves superior results compared to zero-shot CLIP.  

### Challenges
- Adversarial training increases computational complexity and requires high resources.  
- Overfitting issues due to large datasets require better regularization techniques.  

## Installation & Usage

### Prerequisites
- **Python 3.x**
- **PyTorch**
- **Hugging Face Transformers**
- **OpenAI CLIP model**
- **CUDA (for GPU training)**

### Steps to Run
```sh
# Clone the repository
git clone https://github.com/rohitkulkarni08/domain-adaption-clip.git
cd domain-adaption-clip

# Install dependencies
pip install -r requirements.txt

# Run training script
python train.py --dataset flickr30k --epochs 10 --lr 5e-5

# Run evaluation script
python evaluate.py --dataset visual-genome
