# Sentiment Flow – Understanding Twitter Sentiment on Apple and Google Products

## Introduction
In an era where public opinion on products can influence brand perception, companies increasingly rely on sentiment analysis to capture real-time customer feedback. This project aims to classify Twitter sentiment related to Apple and Google products. By using sentiment polarity classification, this analysis seeks to provide actionable insights into customer satisfaction and emerging issues, helping brands like Apple and Google make data-driven decisions regarding product improvements, customer support strategies, and marketing efforts.

## Business Understanding
This project focuses on classifying the sentiment of tweets concerning Apple and Google products. Through sentiment polarity classification, we aim to provide insights that aid in understanding customer satisfaction, identifying issues, and informing strategic responses.

### 1. Problem Statement: 
We aim to classify tweet sentiments as positive, negative, or neutral, providing insights into public perception of Apple and Google products.

### 2. Audience: 
The primary stakeholders are the product, marketing, and customer support teams at Apple and Google.

### 3. Impact: 
The sentiment analysis results will enable Apple and Google to better understand customer opinions, proactively address issues, and enhance product development and customer engagement.

## Data Understanding 
The dataset, sourced from CrowdFlower via data.world, consists of tweets where contributors evaluated the sentiment expressed toward different brands and products. Each tweet is annotated based on the sentiment (positive, negative, or neutral) and, where applicable, the specific brand or product the sentiment is directed at.

## Data Suitability
This dataset is well-suited to our project for several reasons:

1. Relevance: The data focuses on sentiments toward brands and products, aligning closely with our objective of analyzing Twitter sentiment specifically for Apple and Google products.

2. Real-World Context: The tweets reflect genuine user opinions, adding practical value to our sentiment analysis project.

3. Multiclass Capability: With sentiment labels as positive, negative, and neutral, the dataset enables us to develop both binary (positive/negative) and multiclass (positive/negative/neutral) classification models.

3. Dataset Composition
The dataset includes over 9,000 labeled tweets, allowing for a comprehensive exploration of sentiment distribution and linguistic patterns.

## Feature Selection
The tweet content is our primary feature, while the sentiment and target brand/product labels are essential for building and evaluating our classifiers.

## Modelling
These steps facilitate machine learning algorithms to process the emotion variable, convert text into a numerical format for better analysis, ensure the model is not biased towards the majority class, and provide clear metrics to evaluate performance on unseen data.

### Preprocessing 

Prepare data for modeling by:

1. Label Encoding: Converted emotion labels into numerical values.
2. Vectorization: Used TF-IDF and CountVectorizer to transform text data into numerical vectors.
3. SMOTE: Applied SMOTE to handle class imbalance.
4. Train test split: To split the data

### Models

The machine learning algorithims used in this section are:

1. RandomForest
2. Naive Bayes(MultinomialNB)
3. LogisticRegression
4. DecisionTrees

We will use the split data to predict which model will achieve the highest accuracy and use it for deployment.

## Results

### Random forest classifier
**Count Vectorisation Results**

    Best Random Forest Model (Count Vectorization):
    RandomForestClassifier(n_estimators=200, random_state=42)

    Test Accuracy (Count Vectorization): 0.706

    Test Recall (Count Vectorization): 0.705
    
**TFIDF Vectorisation Results**

    Best Random Forest Model (TFIDF Vectorization):
    RandomForestClassifier(random_state=42)

    Test Accuracy (TFIDF Vectorization): 0.837

    Test Recall (TFIDF Vectorization): 0.836

- There is a significant improvement in test accuracy from 0.477 to 0.706 in the model using Count Vectorization.

- We can note an indication that TF-IDF provides a superior feature representation for the Random Forest model.

- The accuracy score is at 84% which is an improvement from 73.3%. The models improvement is due to tuning

### Naive Bayes / (MultinomialNB) model
**Count Vectorisation Results**

    Best Mnb Model (Count Vectorization):
    MultinomialNB(alpha=0.01)

    Test Accuracy (Count Vectorization): 0.660

    Test Recall (Count Vectorization): 0.659

**TFIDF Vectorisation Results**

Best Mnb Model (TFIDF Vectorization):
MultinomialNB(alpha=0.01)

Test Accuracy (TFIDF Vectorization): 0.795

Test Recall (TFIDF Vectorization): 0.795

- The accuracy score is at 79.8% which is an improvement from 76.7%. 

- Note the improvement from 0.66 to 0.798 for the model using TF-IDF Vectorization

### Logistic Regression
**Count Vectorisation Results**

    Best Logistic Regression Model (Count Vectorization):
    LogisticRegression(C=31.0)

    Test Accuracy (Count Vectorization): 0.707

    Test Recall (Count Vectorization): 0.705

**TFIDF Vectorisation Results**

    Best Logistic Regression Model (TFIDF Vectorization):
    LogisticRegression(C=31.0, max_iter=150)

    Test Accuracy (TFIDF Vectorization): 0.831

    Test Recall (TFIDF Vectorization): 0.830

- Count Vectorization based model improved in test accuracy from 0.705 to 0.706

- TF-IDF Vectorization-based model improved from 0.805 to 0.828 after hyperparameter tuning.

- Further indication that the TF-IDF vectorisation is better

### Decison Tree

**Count Vectorisation Results**

Best Decision Tree Model (Count Vectorization):
DecisionTreeClassifier(max_features=5, min_samples_split=5)

Test Accuracy (Count Vectorization): 0.695

Test Recall (Count Vectorization): 0.693

**TFIDF Vectorisation Results**

Best Decision Tree Model (TFIDF Vectorization):
DecisionTreeClassifier(max_features=5, min_samples_split=4)

Test Accuracy (TFIDF Vectorization): 0.758

Test Recall (TFIDF Vectorization): 0.757

- Count Vectorization based model reduced accuracy performance from 0.69 to 0.68.

- TF-IDF Vectorization-based model increased accuracy performance from 0.75 to 0.76.

- Further indication that the TF-IDF vectorisation is better

## Deployment

