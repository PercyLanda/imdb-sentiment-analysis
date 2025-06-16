# 📘 Sentiment Analysis on IMDb Reviews: Summary of Experiments and Models

This document summarizes a series of notebooks exploring sentiment analysis on the IMDb movie review dataset. Across seven stages, we progress from exploratory data analysis to traditional machine learning, and then to deep learning techniques using both Scikit-learn and modern transformer-based models such as BERT. Each notebook builds upon the previous, increasing in modeling complexity, training strategy, and predictive performance. The final section presents a **consolidated comparison table** of all models evaluated, sorted by **test accuracy**.

---

## 1. 🧪 Exploratory Data Analysis (EDA)

**Notebook:** `1-Sentiment-Analysis-EDA.ipynb`

This notebook introduces the IMDb dataset and explores its statistical characteristics and structure before modeling begins.

**Key steps and findings:**

* **Dataset Loading**: 50,000 labeled reviews, evenly split between positive and negative sentiments.
* **Review Lengths**: Vary significantly—from just **4 to 2,470 words**.
* **Outlier Detection**: Identified using:

  * Interquartile Range (IQR)
  * Standard Deviation (±3σ)
  * Boxplots
* **Distribution Analysis**:

  * Multiple histogram binning strategies used: fixed-width, auto, Freedman–Diaconis, and Sturges' Rule.
  * A **Kernel Density Estimate (KDE)** was plotted for a smoothed word count distribution.

> This EDA provides a robust foundation for understanding the dataset’s shape and variability, informing feature engineering and preprocessing steps used in later notebooks.

---

## 2. 🤖 Baseline Model Comparison (Scikit-learn)

**Notebook:** `2-Sentiment-Analysis-Scikit-Learn-Four-Model-Comparison.ipynb`

This notebook evaluates **four classic machine learning classifiers** on the IMDb dataset using a consistent TF-IDF vectorization pipeline.

**Workflow:**

* **TF-IDF Preprocessing**: Up to 10,000 features; unigram and bigram support.
* **Models Evaluated**:

  * Logistic Regression
  * Linear SVM
  * Multinomial Naive Bayes
  * Random Forest
* **Results**:

  * Logistic Regression performed best, followed closely by Linear SVM.
  * Naive Bayes served as a strong baseline.
  * Random Forest underperformed relative to others.
* **Visualization**: A bar chart compared model accuracies.

> These models provided quick and interpretable baselines, with Logistic Regression setting a strong initial benchmark.

---

## 3. 🔍 Tuned Logistic Regression (Scikit-learn)

**Notebook:** `3-Sentiment-Analysis-Scikit-Tuned-Logistic-Regression.ipynb`

Here, we focus solely on **Logistic Regression**, optimizing it with a **Grid Search** over hyperparameters.

**Pipeline**:

* Combined TF-IDF vectorization and Logistic Regression into a single `Pipeline`.
* **GridSearchCV** tuned:

  * `max_features` in TF-IDF
  * `ngram_range`
  * Regularization strength `C` in Logistic Regression

**Best Model Configuration**:

* `max_features=10,000`, `ngram_range=(1,2)`, `C=1`

**Evaluation**:

* **Test Accuracy**: 90.00%
* **ROC AUC**: 0.9646
* Classification report and confusion matrix provided.
* Top positive/negative words identified from model coefficients.
* **Bonus**: Included an interactive command-line review classifier.

> This model matches the baseline logistic regression’s performance while offering higher interpretability and optimized decision boundaries.

---

## 4. 🧠 Tuned Neural Network (TensorFlow + KerasTuner)

**Notebook:** `4-Sentiment-Analysis-tensorflow-KerasTuner.ipynb`

Introduces a **lightweight neural network**, optimized using **KerasTuner’s Hyperband**.

**Key components**:

* **TextVectorization Layer**: Converts raw text to integer sequences.
* **Model Architecture**:

  * Embedding layer (tunable dimension)
  * GlobalAveragePooling
  * Dense and Dropout layers
* **Tuning Strategy**:

  * Tuned dropout, embedding dimension, dense units, and learning rate.

**Best Model**:

* Embedding Dim: 64, Units: 64, Dropout: 0.3, Learning Rate: 1e-3

**Test Accuracy**: 87.77%

> This compact neural network achieved strong generalization while remaining computationally efficient, making it a good candidate for deployment in resource-constrained environments.

---

## 5. 🛠️ Logistic Regression in PyTorch (BoW)

**Notebook:** `5-Sentiment-Analysis-PyTorch-Logistic-Regression.ipynb`

A reproduction of logistic regression using **PyTorch** with a **manual bag-of-words pipeline**.

**Key points**:

* Tokenized reviews and built a vocabulary of the top 10,000 words.
* Manually vectorized inputs using frequency counts.
* Used `nn.Linear` with `CrossEntropyLoss` and Adam optimizer.
* **Evaluation**:

  * **Test Accuracy**: 90.25%
  * **ROC AUC**: 0.9605
* **Interactive CLI**: Implemented for real-time sentiment predictions.

> Despite its simplicity, this PyTorch implementation rivals the performance of more complex pipelines, offering flexibility and transparency.

---

## 6a. 🤏 DistilBERT (Lower Accuracy Version)

**Notebook:** `6a-Sentiment-Analysis-distilBERT-lower-acc.ipynb`

This version of DistilBERT underwent extra fine-tuning but underperformed compared to its sibling model.

**Key Optimizations Made**:

* Increased epochs (2 → 4)
* Batch size (8 → 16)
* Dropout (0.2 → 0.3)
* Mixed-precision enabled for Apple Silicon
* Updated optimizer for Keras 3 compatibility

**Training Overview**:

* Gradual improvement across 4 epochs
* Final Training Accuracy: **80.59%**
* Final Validation Accuracy: **80.80%**
* Final Validation Loss: **0.4995**
* Total Training Time: \~1 hour per epoch (varied between 16 and 28 minutes per 1000 steps)

**Final Results**:

* **Test Accuracy**: 82%
* **F1-Score**: 0.82
* Training time increased, but performance declined.

> Despite more training and regularization, this model lagged behind, suggesting that architectural configuration and learning schedules are more critical than raw compute.

---

## 6b. ✅ DistilBERT (Higher Accuracy Version)

**Notebook:** `6b-Sentiment-Analysis-distilBERT-higher-acc.ipynb`

This run produced significantly better results than the previous DistilBERT version.

**Details**:

* Only 2 training epochs
* Double the training steps per epoch (4000 vs. 2000)
* Well-balanced F1-scores and precision/recall
* Much lower training and validation losses from the start

**Training Overview**:

* Final Training Accuracy: **85.67%**
* Final Validation Accuracy: **85.89%**
* Final Validation Loss: **0.3320**
* Training Loss dropped from **0.4677 → 0.3359**

**Final Performance**:

* **Test Accuracy**: 86%
* **F1-Score**: 0.86
* **Loss**: \~0.33

> Despite shorter training, this version’s careful tuning led to more stable convergence and better generalization.

---

## 7. 🧬 BERT (TFBertForSequenceClassification)

**Notebook:** `7-Sentiment-Analysis-Keras-BERT.ipynb`

This notebook fine-tunes a full **BERT-base-uncased** model using TensorFlow and Keras.

**Pipeline**:

* Used Hugging Face `BertTokenizer` for preprocessing.
* Fine-tuned using early stopping (patience = 2).
* Training stopped at epoch 4 based on validation loss.

**Evaluation**:

* **Test Accuracy**: **92%**
* Precision, recall, and F1-score ≈ 0.92
* Balanced confusion matrix

> This was the best-performing model overall, leveraging BERT's contextual embeddings to achieve state-of-the-art results on this task.

---

## 📊 Consolidated Model Performance Table (Sorted by Accuracy)

| Model                           | Accuracy | Brief Note                                                         |
| ------------------------------- | -------- | ------------------------------------------------------------------ |
| **BERT**                        | 0.9200   | Early stopping at epoch 4; strong balanced results                 |
| **Logistic Regression**         | 0.9028   | Best Scikit-learn model; high-performing baseline                  |
| **Logistic Regression (BoW)**   | 0.9025   | Classic bag-of-words with PyTorch; strong baseline                 |
| **Logistic Regression (Tuned)** | 0.9000   | GridSearchCV-tuned; high ROC AUC (0.9646)                          |
| **Linear SVM**                  | 0.8972   | Runner-up in Scikit-learn comparison                               |
| **DistilBERT (High Acc)**       | 0.8600   | Fewer epochs, better convergence; balanced F1-scores               |
| **Naive Bayes**                 | 0.8684   | Good probabilistic baseline; lightweight and fast                  |
| **Neural Network (Tuned)**      | 0.8777   | Lightweight model tuned via KerasTuner; stable performance         |
| **Random Forest**               | 0.8580   | Least effective traditional model; likely overfit on sparse TF-IDF |
| **DistilBERT (Low Acc)**        | 0.8200   | Extra tuning and epochs did not improve performance                |

---
