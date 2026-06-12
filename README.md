# Jordanian Official Gazette — Part 3: Deep NLP Classification & Semantic Generation

An advanced, end-to-end Deep Learning and Natural Language Processing (NLP) pipeline designed to classify complex Arabic legal documents from the Jordanian Official Gazette. This repository builds upon the cleaned OCR data to perform robust multi-class/multi-label classification and highly accurate semantic tag generation.

---

## 🧠 System Architecture & Pipeline Steps

This repository is structured sequentially to demonstrate a complete, state-of-the-art machine learning lifecycle—from data balancing to hybrid ensemble modeling and sequence generation.

### `01_data_augmentation.py` | Data Balancing & NLP Augmentation
Addresses severe class imbalance in the legal dataset.
* **Contextual Augmentation:** Utilizes AraBERT Masked Language Modeling (MLM) to synthetically generate semantically valid legal phrasing.
* **Strategic Downsampling:** Reduces the dominance of overrepresented sectors to prevent model bias.
* **Data Sanitization:** Implements strict entity-preserving cleaning, ensuring that all augmented data is free from noise and domain-specific stop words.

### `02_hard_negatives_training.py` | Transformer Fine-Tuning & Hard Negative Mining
The core training module for state-of-the-art Arabic Language Models (AraBERT v2, MARBERT, CAMeLBERT).
* **Hard Negative Mining:** Dynamically identifies and forces the model to learn from the most confusing document boundaries and edge cases.
* **Focal Loss Integration:** Applies an adaptive Gamma penalty to focus the optimizer on misclassified and difficult examples.
* **Advanced Scheduling:** Utilizes Cosine Annealing with Warmup to stabilize transformer fine-tuning.

### `03_weak_classes_specialist.py` | Minority Sector Specialist Model
A highly targeted training script designed to rescue underperforming classes.
* Trains a standalone "Specialist Model" exclusively on weak sectors (e.g., Infrastructure, Labor, Trade) that suffer from low F1-scores in the main models.
* Acts as an expert advisor for edge-case legal categories.

### `04_basic_ensemble_models.py` | Foundational Ensemble Learning
Aggregates the predictions of the three main transformer models to boost overall confidence.
* Implements Simple Averaging, Weighted Averaging (based on individual F1 performance), and Stacking using a Logistic Regression Meta-Learner.

### `05_advanced_hybrid_ensemble.py` | Advanced Hybrid Architecture
The pinnacle of the classification pipeline, achieving peak F1-Macro scores.
* Intelligently routes predictions by combining the top-tier generic models (MARBERT/AraBERT) with the Weak Specialist Model. 
* Resolves conflicts by trusting the Specialist Model when predicting rare legal sectors, smoothing out false positives and false negatives simultaneously.

### `06_semantic_generation_and_fallback.py` | AraT5 Generation & Retrieval Fallback
A sophisticated semantic generation engine that moves beyond classification to generate descriptive legal tags.
* **Seq2Seq Generation:** Fine-tunes `AraT5` to auto-generate contextually accurate semantic text summaries for each Gazette entry.
* **Intelligent Fallback Mechanism:** Features a robust retrieval-based safety net. If the AraT5 generation confidence (semantic correlation score) drops below a strict threshold, the system automatically triggers a dictionary-based retrieval fallback to ensure 100% legal accuracy and zero hallucination.

---

## 💻 Tech Stack & Frameworks
* **Core Machine Learning:** PyTorch, Scikit-Learn
* **NLP & Transformers:** HuggingFace `transformers` (AraBERT, MARBERT, CAMeLBERT, AraT5)
* **Data Processing:** Pandas, NumPy
* **Optimization:** Focal Loss, Hard Negative Mining, Adaptive Ensemble Stacking
