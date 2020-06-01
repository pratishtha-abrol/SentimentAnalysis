# SentimentAnalysis
Logistic Regression: A sentiment analysis case study

## Key Concepts
* Build and employ a logistic regression classifier using scikit-learn
* Clean and pre-process data
* Perform feature extraction with nltk
* Tune model hyperparameters and evaluate model accuracy

## Dataset
IMDB movie reviews dataset
http://ai.stanford.edu/~amaas/data/sentiment
Contains 25000 positive and 25000 negative reviews 
Contains at most reviews per movie
At least 7 stars out of 10 : positive (label = 1)
At most 4 stars out of 10 : negative (label = 0)
50/50 train/test split
Evaluation accuracy

## Features: bag of 1-grams with TF-IDF values:
Extremely sparse feature matrix - close to 97% are zeros

## Model: Logistic regression

p(y=1|x)=σ(w.Tx)

Linear classification model
Can handle sparse data
Fast to train
Weights can be interpreted 