# Sentiment Flow – Understanding Twitter Sentiment on Apple and Google Products

## Introduction
In an era where public opinion on products can influence brand perception, companies increasingly rely on sentiment analysis to capture real-time customer feedback. This project aims to classify Twitter sentiment related to Apple and Google products. By using sentiment polarity classification, this analysis seeks to provide actionable insights into customer satisfaction and emerging issues, helping brands like Apple and Google make data-driven decisions regarding product improvements, customer support strategies, and marketing efforts.

## Business Understanding
This project focuses on classifying the sentiment of tweets concerning Apple and Google products. Through sentiment polarity classification, we aim to provide insights that aid in understanding customer satisfaction, identifying issues, and informing strategic responses.

#### 1. Problem Statement: 
We aim to classify tweet sentiments as positive, negative, or neutral, providing insights into public perception of Apple and Google products.

#### 2. Audience: 
The primary stakeholders are the product, marketing, and customer support teams at Apple and Google.

#### 3. Impact: 
The sentiment analysis results will enable Apple and Google to better understand customer opinions, proactively address issues, and enhance product development and customer engagement.

## Data Understanding 
The dataset, sourced from CrowdFlower via data.world, consists of tweets where contributors evaluated the sentiment expressed toward different brands and products. Each tweet is annotated based on the sentiment (positive, negative, or neutral) and, where applicable, the specific brand or product the sentiment is directed at.

### Data Suitability
This dataset is well-suited to our project for several reasons:

1. Relevance: The data focuses on sentiments toward brands and products, aligning closely with our objective of analyzing Twitter sentiment specifically for Apple and Google products.

2. Real-World Context: The tweets reflect genuine user opinions, adding practical value to our sentiment analysis project.

3. Multiclass Capability: With sentiment labels as positive, negative, and neutral, the dataset enables us to develop both binary (positive/negative) and multiclass (positive/negative/neutral) classification models.

3. Dataset Composition
The dataset includes over 9,000 labeled tweets, allowing for a comprehensive exploration of sentiment distribution and linguistic patterns.


### Feature Selection
The tweet content is our primary feature, while the sentiment and target brand/product labels are essential for building and evaluating our classifiers.