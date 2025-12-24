# DeepFakeArt Siamese Network Project Summary

## Overview
This project explores AI-based art forgery detection using Siamese neural networks with two different backbone architectures: ResNet (ResNet-18/34) and DenseNet (DenseNet-121). Both models are trained and evaluated on a dataset of paired images (similar/forgery and dissimilar pairs) with extensive data augmentation and performance analysis.

---

## 1. Data Preparation & Loading
- **Data Source:**
  - Images are organized in `similar` and `dissimilar` folders, with JSON files specifying image pairs for training and validation.
- **Pair Loading:**
  - Custom functions parse JSON files, join root paths, and load image pairs for each category (e.g., Inpainting, Style Transfer, Adversarial, CutMix).
  - Category-wise statistics and breakdowns are printed for transparency.
- **Augmentation:**
  - Training: Resize, random horizontal flip, random rotation, color jitter, normalization.
  - Validation: Resize, normalization.
- **Dataset & DataLoader:**
  - Custom `SiameseDataset` class loads and transforms image pairs.
  - DataLoaders use persistent workers, pin memory, and custom collate functions to handle missing/corrupt images.

---

## 2. Model Architectures
### A. ResNet-based Siamese Networks
- **Backbones:**
  - ResNet-18 (baseline) and ResNet-34 (final, with Dropout regularization).
- **Encoder:**
  - Pretrained ResNet (last FC layer replaced by custom embedding layer with BatchNorm and Dropout for ResNet-34).
- **Siamese Head:**
  - Cosine similarity between embeddings.
  - Output is a logit for binary classification (forgery/similar vs. dissimilar).

### B. DenseNet-based Siamese Network
- **Backbone:**
  - DenseNet-121 (pretrained on ImageNet).
- **Encoder:**
  - Final classifier replaced by a linear embedding layer with BatchNorm.
- **Siamese Head:**
  - Cosine similarity, followed by a prediction head (linear layer) to produce logits suitable for `BCEWithLogitsLoss`.
- **Distributed Data Parallel (DDP):**
  - DenseNet model supports multi-GPU training with PyTorch DDP, including rank/world size setup and distributed samplers.

---

## 3. Training & Validation
- **Loss Function:**
  - `BCEWithLogitsLoss` for binary classification.
  - DenseNet model uses `pos_weight` to address class imbalance.
- **Optimizer:**
  - Adam optimizer (with weight decay for ResNet-34).
- **Learning Rate Scheduler:**
  - `ReduceLROnPlateau` (monitors validation accuracy, reduces LR on plateau).
- **Progress Tracking:**
  - Interactive progress bars (tqdm) and detailed epoch/step reporting.
  - Training/validation loss and accuracy tracked per epoch.
- **Model Saving:**
  - Best model (by validation accuracy) and final model/encoder weights are saved.

---

## 4. Inference & Testing
- **Inference Functions:**
  - Functions provided to load a trained model and predict similarity for a given image pair.
  - Interactive and batch inference tests on validation pairs, with visualizations of results.
- **Example Tests:**
  - Both single and multiple-pair inference tests are performed, with results displayed and compared to ground truth.

---

## 5. Evaluation & Analysis
### A. Confusion Matrix Analysis
- **ResNet-18, ResNet-34, DenseNet-121:**
  - Confusion matrices generated for validation set.
  - Visualized using seaborn heatmaps.
  - Key statistics: true positives, false positives, true negatives, false negatives, and class imbalance observations.
  - DenseNet-121 and ResNet-34 show higher accuracy and better balance than ResNet-18.

### B. Training History Visualization
- **Plots:**
  - Loss and accuracy curves for both training and validation sets over epochs.
  - Used to diagnose overfitting and convergence.

### C. Efficiency Metrics
- **Metrics Computed:**
  - Accuracy, precision, recall, F1-score for both classes.
  - Inference time, GPU utilization, model size, and memory usage.
- **Results:**
  - DenseNet-121: ~81% accuracy, model size ~28.6 MB, GPU utilization ~82%.
  - ResNet-34: ~84% accuracy, model size ~85.4 MB, GPU utilization ~87%.

### D. Per-Category Analysis
- **Category-wise Accuracy:**
  - Performance broken down by forgery type (Inpainting, Style Transfer, Adversarial, CutMix, Dissimilar).
  - Bar and pie charts visualize accuracy and test set composition.
  - Best performance on CutMix, weakest on Adversarial.

---

## 6. Key Techniques & Best Practices
- **Data Augmentation:**
  - Aggressive augmentations to improve generalization.
- **Class Imbalance Handling:**
  - Weighted loss for DenseNet, careful reporting for all models.
- **Regularization:**
  - Dropout and weight decay in ResNet-34 to combat overfitting.
- **Distributed Training:**
  - DDP for DenseNet to leverage multiple GPUs.
- **Robustness:**
  - Custom collate functions and error handling for missing/corrupt images.
- **Visualization:**
  - Extensive use of matplotlib/seaborn for result interpretation.

---

## 7. Outputs & Artifacts
- **Saved Models:**
  - Best and final model weights for each architecture.
- **Plots:**
  - Training history, confusion matrices, per-category performance.
- **Console Outputs:**
  - Detailed logs for data loading, training, validation, and inference phases.

---

## 8. Conclusions
- **DenseNet-121 and ResNet-34 Siamese networks are both effective for AI art forgery detection, with ResNet-34 achieving the highest accuracy.**
- **Category-wise analysis reveals strengths and weaknesses, guiding future improvements.**
- **The project demonstrates best practices in deep learning workflow, including data handling, model design, training, evaluation, and interpretability.**
