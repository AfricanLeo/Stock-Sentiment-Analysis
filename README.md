# Stock Market Sentiment Analysis

The [GameStop story](https://www.nbcnews.com/business/business-news/gamestop-reddit-explainer-what-s-happening-stock-market-n1255922) in early 2021 shocked investors, market watchers and regulators alike.  It displayed the power that social media has granted to groups of like-minded people who are able to meet up with relative easy and force their 'will' upon an institution as powerfull as Wall Street by manipulating the **sentiment** about a stock.   

![](images/NLP-Stock-Market.jpg)

## Stock Market Sentiment
Market sentiment refers to the overall attitude of investors toward a particular security or financial market. It is the feeling or tone of a market, or its crowd psychology, as revealed through the activity and price movement of the securities traded in that market ([Investopedia](https://www.investopedia.com/terms/m/marketsentiment.asp)).  In so many words, rising prices indicate a positive, or bullish sentiment, while failing prices indicate a negative or bearish market sentiment. Understanding market sentiment can help investors make buy or sell decisions.

## This project

Although there are technical indicators, like the VIX / fear index to determine **stock market sentiment**, this project attempts to **determine investor sentiment** by analysing **tweets from investors on Twitter**.

**Natural Language Processing**, or **NLP**, is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.  It works by converting text/words into numbers and then using these numbers in a classifier / machine learning (ML) / artificial intelligence (AI)  model to make predictions.

In this project, I will use various NLP techniques to analise a **dataset of stock market sentiment tweets** and produce a solution to **predict sentiment** of investors by analysing what they are tweeting about it.  The [stock market sentiment dataset](https://www.kaggle.com/yash612/stockmarket-sentiment-dataset) is kindly provided on the Kaggle website by [Yash Chaudhary](https://www.kaggle.com/yash612).  The dataset contains approximately 6,000 tweets regarding stocks, trading and economic forecasts from twitter.  Each tweet was classified as having a positive(1) or negative(0) sentiment. 

## Project Approach 

Researching best practices for conducting a sentiment analysis, I came across an [exceptionally comprehensive article](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e) on how to solve NLP problems in 'the real world'.  Although I did not follow all the steps, I largely used  [Emmanuel Ameisen](https://medium.com/@EmmanuelAmeisen)'s structure to plan and execute this project.

The high level approach can be summed up as follows:

1.   Clean and prep the data
2.   Start with a quick and simple classifier
3.   Evaluate and explain it's predictions
4.   Use insights gained in step 3, make changes to the model and repeat. 

## Clean and Preprocess the Data

Before we can convert text to numbers, we need to process the input text and make sure that the text we convert makes the most sense to a model.  In order to analise text like tweets or sentences,  we need to first clean it up by removing unnecessary characters that add noise to the data.  It is also good practice to remove stopwords that are common throughout text but does not necessarily add any additional meaning to the text. 

The next step is called **tokenisation** which is the process of taking a string of words and chopping it up in words, or better said '**tokens**'. 

## Bag of Words

Once we have the tokens or words, we need to somehow convert these into numbers whilst maintaining enough information so that the text can be 'understood'.  One very intuitive way to do this is by building a vocabulary of words that are present in our cleaned text, and use that as a basis to work from 'understanding' the text.

One such approach is the **Bag of Words (BoW)** model which is a representation of a text (sentence / document / tweet) as a list (bag) of all its words, disregarding grammar and word order but keeping count of **how many times** a word is in the input text. 

Once encoded every tweet will be represented as a vector with mostly zero's, but values where words are represented, and indicating how often they are repeated in a tweet. Before we continue, it is good to attempt a visualisation by reducing the dimensions of the data using a technique like PCA (Principal Component Analysis). 

![](images/bow_pca.png

The first look at the data is not too bad.  Although the classes are not perfectly split, we can see at least some seperation between the two classes.  

Next up the data is split into a training- and testing set and these are run through a list of classifiers.  After training the Logistic Regression  model gives an **accuracy score of 80%** which is very good as a first attempt.  

## Inspect the BoW model

Now we use the test data to make predictions using the classifier.  Evaluating the confusion matrix, we can see that the model's worst quadrant is on False-Positives.  As this model is used to make investment decisions, it is 'safer' to have more False-Positives (13%) as opposed to False-Negatives (only 7%) as this will produce more conservative predictions.  

Before we move on, we should first inspect the model to understand which words the model uses to achieve this result.  This figure plots the top ten words for each sentiment, positive and negative.  

![](images/bow_


[**Back to Portfolio**](https://africanleo.github.io/Stock-Sentiment-Analysis/)
