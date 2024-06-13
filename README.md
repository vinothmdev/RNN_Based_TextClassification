# Problem Description and Data

This Challenge is to develop a model to predict whether a tweet describes a real disaster (label 1) or not (label 0) using the provided dataset. The goal is to assist disaster relief organizations and news agencies in programmatically monitoring Twitter for real-time emergency announcements.

## Problem Context

Twitter is a crucial communication channel during emergencies due to the widespread use of smartphones. People often report emergencies as they observe them, making it essential for agencies to distinguish between actual disaster reports and non-disaster content.

## Dataset Origin

The dataset was created by Figure-Eight and initially shared on their 'Data For Everyone' website.

## Evaluation Metric

The performance of the predictions will be evaluated using the F1 score, which balances precision and recall between the predicted and actual labels.

## Key Challenges
### Ambiguity in Tweets
Tweets can be ambiguous and may not clearly indicate whether they describe a real disaster.

### Text Processing

Effective preprocessing of tweet text to handle slang, abbreviations, and varied linguistic styles is critical.

### Imbalanced Data: There may be an imbalance between disaster-related and non-disaster-related tweets, impacting the model's performance.

## Plans to Address the Problem
### Data Preprocessing:

**Clean** the tweet text by removing special characters, URLs, and irrelevant content.

**Normalize** the text by converting it to lowercase and removing stop words.

**Tokenize and lemmatize/stem** the text to standardize the words.

**Feature Engineering** Extract relevant features such as word n-grams, TF-IDF scores, and sentiment analysis scores.

# Model developmetn approach
I build a Logistic Regression base model, with TF-IDF Vectorizer as base model.  Then build a base LSTM based model.  Finally did a hyper parameter turing using Karas Tuner with another version of LSTM.  Then compared results as below.

## Summary of Model Performance

#### Model Evaluation Metrics

| Model                | Accuracy | F1 Score | Hyperparameters                                                                                                                                                     |
|----------------------|----------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression  | 0.795470 | 0.729038 | None                                                                                                                                                                |
| Base LSTM            | 0.764823 | 0.695427 | None                                                                                                                                                                |
| Tuned LSTM           | 0.759494 | 0.708636 | {'embedding_dim': 200, 'num_layers': 1, 'lstm_units': [64], 'dropout_rates': [0.3], 'learning_rate': 0.001}                                                          |

## Key Observations

### Logistic Regression
This model achieved the highest accuracy (0.795470) and the highest F1 score (0.729038) among the three models. This suggests that for this specific dataset and task, Logistic Regression is a very effective baseline model.
  
### Base LSTM
The base LSTM model, without any hyperparameter tuning, showed lower performance compared to Logistic Regression, with an accuracy of 0.764823 and an F1 score of 0.695427. This indicates that a simple LSTM model may not be as effective as Logistic Regression for this task.

### Tuned LSTM
The hyperparameter-tuned LSTM model showed slight improvements in F1 score (0.708636) compared to the base LSTM model, although its accuracy (0.759494) was slightly lower than the base LSTM. The tuned model’s parameters, such as embedding dimension and LSTM units, were optimized, but it still did not surpass the performance of the Logistic Regression model in terms of accuracy.

##### Hyperparameters for the Tuned LSTM
  - Embedding Dimension: 200
  - Number of LSTM Layers: 1
  - LSTM Units: [64] (one layer with 64 units)
  - Dropout Rates: [0.3] (one layer with a dropout rate of 0.3)
  - Learning Rate: 0.001

# Conclusion

Based on results it clearly shows the importance of benchmarking various models and tuning hyperparameters to find the most effective solution for a given machine learning task.

# References


- [Text classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
- [Scikit-Learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TensorFlow v2.16.1 Tokenizer Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
- [Scikit-Lean TfidfVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Medium Article about TF-IDF](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a)
- [TensorFlow Resouce - Tokenizing with TF Text](https://www.tensorflow.org/text/guide/tokenizers)
- [KerasTuner Documentation](https://keras.io/keras_tuner/)
- [Medium Article: Hypertuning a LSTM with Keras Tuner to forecast solar irradiance](https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb)
- [Stock Forecasting with LSTM, part 2 Keras Tuner](https://kamran-afzali.github.io/posts/2022-02-20/tuner.html)
- [TensorFlow Learning Resource: Introduction to the Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)