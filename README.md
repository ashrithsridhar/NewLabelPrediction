# Optimized News Label Prediction

This repository contains the implementation of a document classification task using neural networks.

## Project Overview

The goal of this project is to classify documents in a corpus into one of five categories: sport, business, politics, entertainment, and tech. The implementation involves training various linear models and evaluating each one using 5-fold cross-validation. The best performing model is then used to run inference on a test set to predict the labels.

## Dataset

The dataset used in this project consists of the following:
- **Training Data**: Contains the raw text of news articles with their corresponding categories.
- **Test Data**: Contains the raw text of news articles without labels.

### Training Data Format
- **ArticleId**: Unique identifier for the article.
- **Text**: The content of the article.
- **Category**: The category of the article (sport, business, politics, entertainment, tech).

### Test Data Format
- **ArticleId**: Unique identifier for the article.
- **Text**: The content of the article.

## Implementation Details

The implementation consists of the following steps:

1. **Data Preprocessing**:
    - Preprocess the raw training data by constructing features such as n-grams and keyword extractions.
    - Use `CountVectorizer` as the initial feature extraction method.

2. **Model Training and Evaluation**:
    - Train neural networks with 2 hidden layers, each having 128 neurons.
    - Evaluate the performance using 5-fold cross-validation.
    - Explore different feature extraction methods such as TFIDF and BERT, and evaluate their performance.
    - Report the average training and validation accuracy and their standard deviations for different feature construction methods.

3. **Hyperparameter Tuning**:
    - Tune the learning rates and optimizers using the best performing feature extraction method.
    - Evaluate the performance using 5-fold cross-validation and report the results.

4. **Inference**:
    - Use the best performing model to predict labels for the test data.

## Results

### Feature Extraction Methods

| Feature Method    | Avg Training Accuracy | Std Training Accuracy | Avg Validation Accuracy | Std Validation Accuracy |
|-------------------|-----------------------|-----------------------|-------------------------|-------------------------|
| CountVectorizer   | 99.44%                | 3.59%                 | 96.24%                  | 1.06%                   |
| TFIDF             | 98.60%                | 7.86%                 | 96.99%                  | 0.53%                   |
| BERT              | 98.96%                | 4.97%                 | 98.21%                  | 0.90%                   |

### Hyperparameter Tuning

#### Learning Rates

| Learning Rate | Avg Training Accuracy | Std Training Accuracy | Avg Validation Accuracy | Std Validation Accuracy |
|---------------|-----------------------|-----------------------|-------------------------|-------------------------|
| 0.0001        | 94.87%                | 12.50%                | 98.31%                  | 1.18%                   |
| 0.0003        | 97.74%                | 8.32%                 | 98.02%                  | 1.08%                   |
| 0.001         | 98.89%                | 5.41%                 | 98.31%                  | 0.79%                   |
| 0.003         | 99.14%                | 4.26%                 | 98.49%                  | 0.21%                   |
| 0.01          | 98.88%                | 4.18%                 | 98.50%                  | 0.61%                   |
| 0.03          | 98.01%                | 7.14%                 | 97.65%                  | 1.15%                   |
| 0.1           | 49.36%                | 19.68%                | 53.79%                  | 22.90%                  |

### Optimizers

**Random Forest:**
- Average Train Accuracy: 99.76%
- Average Validation Accuracy: 96.80%

**Gradient Boosting:**
- Average Train Accuracy: 100.00%
- Average Validation Accuracy: 96.71%

## Files

- `code.ipynb`: The notebook containing all the code for the assignment.
- `description.pdf`: Description of the results for all questions.
- `labels.csv`: Predicted labels for the test data.

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/ashrithsridhar/NewsLabelPrediction.git
    ```

2. Navigate to the project directory:
    ```sh
    cd NewsLabelPrediction
    ```

3. Open the Jupyter Notebook:
    ```sh
    jupyter notebook code.ipynb
    ```

4. Run the notebook to execute the code and generate the results.

## Conclusion

This project demonstrates the application of neural networks for document classification, including data preprocessing, feature extraction, model training, hyperparameter tuning, and inference.

