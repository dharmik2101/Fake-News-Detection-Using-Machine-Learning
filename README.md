# Fake News Detection Using Machine Learning

This project is focused on detecting fake news using machine learning techniques. The dataset is preprocessed and analyzed before training a classification model. The code explores various steps, from shuffling the dataset and removing irrelevant text to visualizing word frequency and training different classifiers.

# Overview
This project includes:
- Shuffling the dataset to avoid biases
- Text preprocessing to remove stopwords and punctuation
- Word cloud and frequency bar chart visualizations
- Machine learning models for classification (Logistic Regression and Decision Tree)

## Dataset
   https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/dharmik2101/Fake-News-Detection-Using-Machine-Learning.git
   cd fake-news-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources**:
   Run the script to download stopwords and tokenizer data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Data Preprocessing
- **Shuffle and Reset Index**: Randomizes the dataset to prevent order bias and removes unnecessary index column.
- **Text Preprocessing**: Cleans the text by removing stopwords and punctuation using NLTK. The cleaned text is then ready for model input.

## Exploratory Data Analysis (EDA)
- **Word Cloud Visualization**: Generates separate word clouds for real and fake news to observe common terms.
- **Frequency Bar Chart**: Shows the 20 most common words, aiding insight into word patterns across classes.

## Model Training and Evaluation
1. **Logistic Regression**: Basic linear classifier, with accuracy evaluated on training and test sets.
2. **Decision Tree Classifier**: Uses a decision tree model, often achieving higher accuracy.

**Model Accuracy**:
- Logistic Regression: Training ~99.4%, Test ~98.9%
- Decision Tree: Training ~99.9%, Test ~99.5%

3. **Confusion Matrix**: Visualizes the Decision Tree model's performance in predicting real vs. fake news.

## Results
- Both classifiers demonstrate high accuracy, with Decision Tree outperforming Logistic Regression.
- The model successfully detects fake news with high precision.

## Reference
Geeks for Geeks.

---


