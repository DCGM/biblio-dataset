# 📄 BiblioPage Evaluation Scripts

This repository contains official evaluation scripts for the **BiblioPage** dataset — a collection of 2,118 scanned title pages annotated with structured bibliographic metadata. The dataset serves as a benchmark for bibliographic metadata extraction, document understanding, and visual language model evaluation.

## 🔍 About the Dataset

- 2,118 annotated title pages from Czech libraries (1485–21st century)
- 16 bibliographic attributes (e.g., Title, Author, Publication Date, etc.)
- Annotations include both text and precise bounding boxes
- Evaluation results available for YOLO, DETR, and VLLMs (GPT-4o, LLaMA 3)

You can download the dataset on [Zenodo](https://zenodo.org/records/15683417).  
For more details, see the [BiblioPage paper](https://arxiv.org/abs/2503.19658v1).


## ✅ Evaluation Script

This repo provides:
- Attribute-level evaluation with precision, recall, F1, and mAP
- CER-based string matching with configurable threshold
- Normalization for punctuation, whitespace, and historical characters
- JSON-to-JSON comparison for model predictions vs. ground truth

### Usage

```bash
python TODO
```

## 📜 License

MIT License. For academic use, please cite our paper.
