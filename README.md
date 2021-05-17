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

