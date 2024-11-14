# Sentiment Flow – Understanding Twitter Sentiment on Apple and Google Products
# Business Understanding
## Introduction

In an era where public opinion on products can shape brand perception, companies increasingly rely on Natural Language Processing (NLP) for real-time customer feedback analysis. This project, applies NLP techniques to classify Twitter sentiment related to Apple and Google products, addressing a real-world need for understanding public sentiment in a rapidly evolving market. By using sentiment polarity classification,it provides actionable insights into customer satisfaction and emerging issues, enabling companies, marketing teams, and decision-makers to make data-driven decisions. These insights help brands like Apple and Google improve products, refine customer support strategies, and optimize marketing efforts based on social media sentiment.

## Problem Statement

The problem is to accurately classify the sentiment of tweets related to **Apple and Google products**. We want to determine whether a tweet expresses a positive, negative, or neutral sentiment. This classification can help companies understand customer satisfaction, identify potential issues, and tailor their responses accordingly.

## Stakeholders 

- **The companies:** Apple & Google- Considering these companies are direcly affected by the sentiment, it is important for them to gauge the perception of their products so as to identify the areas of improvement.

- **Marketing team:** this sentiment analysis and model can help them respond to negative feedback, adjust their marketing campaigns and highlight the positive aspects of their products.

- **The customer support teams &decision makers:** the sentiment analysis is important for they can use it to improve product development, customer support and brand reputation.

## Business Value

By accurately classifying tweets, our NLP model provides actionable insights to stakeholders, such as:

    + Identifying negative sentiment can help companies address issues promptly.
    + Recognizing positive sentiment can guide marketing efforts and reinforce successful strategies.
    + Understanding neutral sentiment can provide context and balance.

## Objectives 
### Main Objective

To develop a NLP (Natural Language Processing) multiclass classification model for sentiment analysis, aim to achieve a **recall score of 80%** and an **accuracy of 80%**. The model should categorize sentiments into three classes: **Positive**, **Negative**, and **Neutral**.

### Specific Objectives

- To idenitfy the most common words used in the dataset using Word cloud.

- To confirm the most common words that are positively and negatively tagged.

- To recognize the products that have been opined by the users.

- To spot the distribution of the sentiments.

### Conclusion 

Our NLP model will contribute valuable insights to the real-world problem of understanding Twitter sentiment about Apple and Google products. Stakeholders can leverage this information to enhance their decision-making processes and improve overall customer satisfaction.

## Data Understanding

### Data source

The dataset originates from **CrowdFlower via data.world**. Contributors evaluated tweets related to various brands and products. Specifically:
- Each tweet was labeled as expressing **Positive emotion**, **Negative emotion**, **No emotion toward brand or product** or **I can't tell**, toward a brand or product.
- If emotion was expressed, contributors specified which brand or product was the target.

### Suitability of the Data
Here's why this dataset is suitable for our project:
- **Relevance**: The data directly aligns with our business problem of understanding Twitter sentiment for Apple and Google products.
- **Real-World Context**: The tweets represent actual user opinions, making the problem relevant in practice.
- **Multiclass Labels**: We can build both binary (positive/negative) and multiclass (positive/negative/neutral) classifiers using this data.

### Dataset Size
The dataset contains **over 9,000 labeled tweets**. We'll explore its features to gain insights.

### Descriptive Statistics
- **tweet_text**: The content of each tweet.
- **is_there_an_emotion_directed_at_a_brand_or_product**: No emotion toward brand or product, Positive emotion, Negative emotion, I can't tell
- **emotion_in_tweet_is_directed_at**: The brand or product mentioned in the tweet.

### Feature Selection
**Tweet text** is the primary feature. The emotion label and target brand/product are essential for classification.

### Data Limitations
- **Label Noise**: Human raters' subjectivity may introduce noise.
- **Imbalanced Classes**: We'll address class imbalance during modeling.
- **Contextual Challenges**: Tweets are often short and context-dependent.
- **Incomplete & Missing Data**: Could affect the overall performance of the models.


## 4.Data Cleaning & Feature Engineering

## Modelling

These steps facilitate machine learning algorithms to process the emotion variable, convert text into a numerical format for better analysis, ensure the model is not biased towards the majority class, and provide clear metrics to evaluate performance on unseen data.

### Preprocessing 

Prepare data for modeling by:
1. Label Encoding: Converted emotion labels into numerical values.
2. Vectorization: TF-IDF and CountVectorizer are used to transform text data into numerical vectors.
3. SMOTE: Applied SMOTE to handle class imbalance.
4. Train test split: To split the data

### Models

The machine learning algorithms used in this section are:
1. RandomForest
2. Naive Bayes(MultinomialNB)
3. LogisticRegression
4. Decision Trees

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

- The accuracy score is at 84% an improvement from 73.3%. The models' improvement is due to the tuning

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

- The accuracy score is 79.8%, an improvement from 76.7%. 

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

- Count vectorization-based model improved in test accuracy from 0.705 to 0.706

- TF-IDF Vectorization-based model improved from 0.805 to 0.828 after hyperparameter tuning.

- Further indication that the TF-IDF vectorization is better

### Decision Tree
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

- The count vectorization-based model reduced accuracy performance from 0.69 to 0.68.

- The TF-IDF vectorization-based model increased accuracy performance from 0.75 to 0.76.

- Further indication that the TF-IDF vectorization is better

## Deployment

