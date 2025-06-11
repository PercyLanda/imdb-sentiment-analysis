# 🎬 Sentiment Analysis on IMDb Movie Reviews

This project performs binary sentiment classification (positive or negative) on IMDb movie reviews using both classical machine learning and modern deep learning techniques, including transformer-based models like BERT.

---

## 📌 Problem Statement

Customer reviews contain valuable insights but are time-consuming to analyze manually. This project automates sentiment analysis of movie reviews from IMDb to classify them as either **positive** or **negative**, enabling scalable insights into user feedback.

---

## 📂 Dataset

- **Source:** [IMDb 50K Movie Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description:** 50,000 balanced positive/negative reviews. Ideal for binary classification tasks.

---

## 🧪 Methodology

### Preprocessing

- Tokenization, lowercasing, stopword removal
- Text vectorization using TF-IDF and BERT tokenizer

### Models Used

- ✅ Logistic Regression, Naive Bayes, SVM, Random Forest (scikit-learn)
- ✅ Deep Neural Networks (TensorFlow, PyTorch)
- ✅ Transformer Models: BERT, DistilBERT (Hugging Face, Keras)

### Evaluation Metrics

- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC-AUC

---

## 🚀 Notebooks & Scripts

| Filename                                              | Description                                                   |
|-------------------------------------------------------|---------------------------------------------------------------|
| `Sentiment-Analysis-visualizations.ipynb`             | EDA: Review lengths, sentiment distribution, KDE plots        |
| `Sentiment-Analysis-Scikit-learn-Model-Comparison.ipynb` | Classical model training + accuracy visualization             |
| `Sentiment-Analysis-Scikit-Learn.ipynb`               | TF-IDF + Logistic Regression pipeline with GridSearch         |
| `Sentiment-Analysis-tensorflow-KerasTuner.ipynb`      | TensorFlow NN + KerasTuner for hyperparameter search          |
| `Sentiment-Analysis-Pytorch.ipynb`                    | PyTorch model: Tokenization, training, evaluation             |
| `Sentiment-Analysis-distilBERT-to-fix.ipynb`          | DistilBERT fine-tuning and evaluation                         |
| `Sentiment-Analysis-Keras-BERT.ipynb`                 | BERT with early stopping and classification report            |

---

## 📈 Results

| Model                     | Accuracy | File Name                                         | Notes                                             |
|---------------------------|----------|--------------------------------------------------|--------------------------------------------------|
| Random Forest             | 0.8424   | `4a-Sentiment-Analysis-Scikit-learn-Model-Comparison.ipynb` | Lower accuracy, slower; baseline ensemble model   |
| Naive Bayes               | 0.8586   | `4a-Sentiment-Analysis-Scikit-learn-Model-Comparison.ipynb` | Fast, simple; strong baseline                    |
| Support Vector Machine    | 0.8874   | `4a-Sentiment-Analysis-Scikit-learn-Model-Comparison.ipynb` | Best of classic models                           |
| Logistic Regression (TF-IDF) | 0.8944 | `4b-Sentiment-Analysis-Scikit-learn.ipynb`       | TF-IDF + GridSearchCV optimized                  |
| TensorFlow Neural Net     | 0.9018   | `4c-Sentiment-Analysis-tensorflow-KerasTuner.ipynb` | Tuned via KerasTuner                             |
| PyTorch Deep Learning     | 0.9112   | `4d-Sentiment-Analysis-Pytorch.ipynb`            | Custom tokenization + training loop              |
| DistilBERT (Hugging Face) | 0.9169   | `4f-Sentiment-Analysis-distilBERT-to-fix.ipynb`  | Lightweight transformer, solid results           |
| BERT (Keras)              | 0.9230   | `4g-Sentiment-Analysis-Keras-BERT.ipynb`         | Best performance; early stopping after epoch 4   |

---

## 🔍 Conclusion

- BERT-based models outperform classical methods but require more computational resources.
- Simpler models like Logistic Regression offer good accuracy and interpretability.
- The project demonstrates how sentiment analysis can be scaled using both traditional and modern NLP techniques.

---

## 📚 Future Work

- Multiclass sentiment classification (e.g., 1–5 stars)
- Real-time sentiment detection
- Multilingual sentiment analysis

---

## 🧰 Tools & Libraries

- Python, scikit-learn, TensorFlow, PyTorch  
- Hugging Face Transformers, Keras, KerasTuner  
- Matplotlib, Seaborn for visualizations

---

## 🗂️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/PercyLanda/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
