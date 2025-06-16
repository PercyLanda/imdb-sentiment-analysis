# 🎬 Sentiment Analysis on IMDb Movie Reviews

This project performs binary sentiment classification (positive or negative) on IMDb movie reviews using both classical machine learning and modern deep learning techniques, including transformer-based models like BERT.

---

## 📌 Problem Statement

Customer reviews contain valuable insights but are time-consuming to analyze manually. This project automates sentiment analysis of movie reviews from IMDb to classify them as either **positive** or **negative**, enabling scalable insights into user feedback.

---

## 📂 Dataset

* **Source:** [IMDb 50K Movie Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Description:** 50,000 balanced positive/negative reviews. Ideal for binary classification tasks.

---

## 🧪 Methodology

### Preprocessing

* Tokenization, lowercasing, stopword removal
* Text vectorization using TF-IDF and BERT tokenizer

### Models Used

* ✅ Logistic Regression, Naive Bayes, SVM, Random Forest (scikit-learn)
* ✅ Deep Neural Networks (TensorFlow, PyTorch)
* ✅ Transformer Models: DistilBERT, BERT (Hugging Face, Keras)

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* ROC-AUC

---

## 🚀 Notebooks & Scripts

| Filename/Notebook                                               | Short Description                                                                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `1-Sentiment-Analysis-EDA.ipynb`                                | Introduces the IMDb dataset and explores its statistical characteristics and structure before modeling begins.            |
| `2-Sentiment-Analysis-Scikit-Learn-Four-Model-Comparison.ipynb` | Evaluates four classic machine learning classifiers on the IMDb dataset using a consistent TF-IDF vectorization pipeline. |
| `3-Sentiment-Analysis-Scikit-Tuned-Logistic-Regression.ipynb`   | Focuses solely on Logistic Regression, optimized via Grid Search over hyperparameters.                                    |
| `4-Sentiment-Analysis-tensorflow-KerasTuner.ipynb`              | Introduces a lightweight neural network, optimized using KerasTuner’s Hyperband.                                          |
| `5-Sentiment-Analysis-PyTorch-Logistic-Regression.ipynb`        | Reimplements logistic regression using PyTorch with a manual bag-of-words pipeline.                                       |
| `6a-Sentiment-Analysis-distilBERT-lower-acc.ipynb`              | DistilBERT model with additional fine-tuning that underperformed relative to its counterpart.                             |
| `6b-Sentiment-Analysis-distilBERT-higher-acc.ipynb`             | DistilBERT model variant that achieved higher accuracy through more effective fine-tuning.                                |
| `7-Sentiment-Analysis-Keras-BERT.ipynb`                         | Fine-tunes a full BERT-base-uncased model using TensorFlow and Keras.                                                     |

---

## 📈 Results

### 📊 Consolidated Model Performance Table (Sorted by Accuracy)

| Model                       | Accuracy | Brief Note                                                     |
| --------------------------- | -------- | -------------------------------------------------------------- |
| **BERT**                    | 0.9230   | Best overall; strong balanced performance; early stopping used |
| Logistic Regression         | 0.9028   | Best performer among classic models (TF-IDF + Scikit-learn)    |
| Logistic Regression (BoW)   | 0.9025   | PyTorch implementation using bag-of-words; robust baseline     |
| Logistic Regression (Tuned) | 0.9000   | GridSearchCV optimized; high ROC AUC (0.9646)                  |
| Linear SVM                  | 0.8972   | Runner-up among traditional models                             |
| Neural Network (Tuned)      | 0.8777   | Lightweight model via KerasTuner; good generalization          |
| Naive Bayes                 | 0.8684   | Simple baseline using TF-IDF                                   |
| distilBERT (Higher Acc)     | 0.8600   | Fewer epochs, better convergence than lower-accuracy variant   |
| Random Forest               | 0.8580   | Underperformed relative to others                              |
| distilBERT (Lower Acc)      | 0.8200   | More training but underperformed; suggests diminishing returns |

---

## 🔍 Conclusion

* BERT-based models outperform classical methods but require more computational resources.
* Logistic Regression continues to offer a strong balance of accuracy and simplicity.
* The project demonstrates how sentiment analysis can be scaled using both traditional and modern NLP techniques, depending on resource availability and desired accuracy.

---

## 📚 Future Work

* Multiclass sentiment classification (e.g., 1–5 stars)
* Real-time sentiment detection
* Multilingual sentiment analysis

---

## 🧰 Tools & Libraries

* Python, scikit-learn, TensorFlow, PyTorch
* Hugging Face Transformers, Keras, KerasTuner
* Matplotlib, Seaborn for visualizations

---

## 🗂️ How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/PercyLanda/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

---
