#!/usr/bin/env python
# coding: utf-8

# # SOCIAL MEDIA SENTIMENT ANALYSIS:
Sentiment analysis(opinion mining).
A technique aimen at obtaining the subjective opinion expressed in text, video, or audio data.
It allows businesses to understand better how their stakeholders feel in a particular situation. 
# Starting the Sentiment Analysis of App Reviews: Importing Python Libraries and Loading the Dataset

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load dataset

# In[4]:


linkedin_data = pd.read_csv("linkedin-reviews.csv")


# Display the first rows of the dataset

# In[5]:


print(linkedin_data.head())


# The dataset features two key columns: Review and Rating. The Review column contains textual feedback, while the Rating column holds the associated numerical scores.

# Let's delve into the details of the columns:

# In[6]:


print(linkedin_data.info())


# # Exploratory Data Analysis(EDA)

# Step-by-Step Data Exploration: Analyzing Ratings Distribution and Uncovering Review Insights

# Let's start by examining how the ratings are distributed:

# # Plotting the distribution of ratings:

# In[9]:


sns.set(style = "whitegrid")
plt.figure(figsize =(9,5))
sns.countplot(data=linkedin_data, x = 'Rating')
plt.title('Distribution of Ratings')
plt.xlabel ('Rating')
plt.ylabel('Count')
plt.show()


# Here's a look at the ratings distribution from the LinkedIn reviews dataset. This visualization clearly shows how many reviews fall into each rating category, ranging from 1 to 5.

# Next, we'll delve into the length of the reviews. This analysis can reveal correlations with sentiment or the level of detail provided in the feedback. We'll start by calculating the length of each review and then move on to visualizing this data.

# In[10]:


# Calculating the length of each review
linkedin_data['Review Length'] = linkedin_data['Review'].apply(len)

# Plotting the distribution of review lengths
plt.figure(figsize=(9, 6))
sns.histplot(linkedin_data['Review Length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Length of Review')
plt.ylabel('Count')
plt.show()


# Adding Annotations to:The annotation in the provided code example highlights the mean length of the reviews.

# In[11]:


plt.figure(figsize=(9, 6))
sns.histplot(linkedin_data['Review Length'], bins=50, kde=True, color='salmon')
plt.title('Distribution of Review Lengths')
plt.xlabel('Length of Review')
plt.ylabel('Count')
plt.axvline(x=linkedin_data['Review Length'].mean(), color='red', linestyle='--')
plt.text(linkedin_data['Review Length'].mean() + 10, 50, 'Mean Length', color='red')
plt.show()


# # Incorporating Sentiment Labels into the Dataset.

# The next step involves labeling the data with sentiments using TextBlob. TextBlob assigns a polarity score to text, ranging from -1 (very negative) to 1 (very positive). We can use these scores to categorize each review's sentiment as positive, neutral, or negative. To get started, you can install TextBlob by running the following pip command in your terminal or command prompt:

# In[12]:


pip install textblob


# Let's move forward with labeling the dataset for sentiment analysis using TextBlob. This tool will help us classify each review based on its polarity score.
# We will start by importing the necessary libraries!

# In[13]:


from textblob import TextBlob

def textblob_sentiment_analysis(review):
    # Analyzing the sentiment of the review
    sentiment = TextBlob(review).sentiment
    # Classifying based on polarity
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Applying TextBlob sentiment analysis to the reviews
linkedin_data['Sentiment'] = linkedin_data['Review'].apply(textblob_sentiment_analysis)

# Displaying the first few rows with the sentiment
print(linkedin_data.head())


# The dataset now features sentiment labels for each review, categorizing them as Positive, Negative, or Neutral according to the polarity scores provided by TextBlob.

# # App Reviews Sentiment Analysis

# With our dataset now labeled, let's dive into the sentiment analysis of app reviews. We'll start by examining the distribution of sentiments throughout the dataset, providing us with an overview of the general sentiment trends in the reviews.

# In[14]:


# Analyzing the distribution of sentiments
sentiment_distribution = linkedin_data['Sentiment'].value_counts()

# Plotting the distribution of sentiments
plt.figure(figsize=(9, 5))
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# Based on the sentiment analysis of the app reviews dataset:
# 
# Positive Sentiment Dominance: The majority of the reviews are classified as Positive, indicating that most users have a favorable opinion about the app.
# 
# Substantial Neutral Sentiment: A significant number of reviews fall into the Neutral category, suggesting that many users have mixed or indifferent feelings about the app.
# 
# Lower Negative Sentiment: The Negative sentiment category has the least number of reviews, highlighting areas that may require improvement but are less common compared to positive and neutral feedback.
# 
# These insights provide a clear overview of the general sentiment trends within the app reviews, helping to identify strengths and areas for potential enhancement.

# # Next, we'll delve into the relationship between sentiments and ratings. This analysis will help us determine if there's a correlation between the sentiment expressed in the text and the numerical ratings. We'll examine how sentiments are distributed across various rating levels to uncover any patterns or insights.

# In[17]:


plt.figure(figsize=(10, 5))
sns.countplot(data=linkedin_data, x='Rating', hue='Sentiment', palette='Set1')
plt.title('Sentiment Distribution Across Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()


# # Next, let's dive into text analysis to uncover common words or themes within each sentiment category. This will involve generating word clouds to visualize the most frequently occurring words in positive, negative, and neutral reviews.

# In[20]:


pip install wordcloud


# In[21]:


from wordcloud import WordCloud

# Function to generate word cloud for each sentiment
def generate_word_cloud(sentiment):
    text = ' '.join(review for review in linkedin_data[linkedin_data['Sentiment'] == sentiment]['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.axis('off')
    plt.show()

# Generating word clouds for each sentiment
for sentiment in ['Positive', 'Negative', 'Neutral']:
    generate_word_cloud(sentiment)


# And that's how you can conduct Reviews Sentiment Analysis using Python!

# App Reviews Sentiment Analysis is an invaluable resource for app developers and businesses, enabling them to gain insights from user feedback, prioritize updates, and foster a positive user community. By employing data analysis techniques, it's possible to classify reviews into positive, negative, or neutral sentiments. I hope you found this article on App Reviews Sentiment Analysis using Python helpful.

# Summary of LinkedIn Reviews Sentiment Analysis
# Positive Sentiment Prevalence: Most reviews are positive, indicating high user satisfaction and favorable feedback about LinkedIn.
# 
# Significant Neutral Feedback: A considerable number of reviews are neutral, suggesting that many users have mixed or indifferent feelings about certain aspects of the platform.
# 
# Negative Sentiment Insights: Although fewer in number, negative reviews are crucial for identifying areas needing improvement.
# 
# Sentiment and Ratings Correlation: Higher ratings are generally associated with positive sentiments, while lower ratings correlate with neutral or negative sentiments.
# 
# Common Themes: Text analysis identified common words and themes within each sentiment category, helping to understand user priorities and recurring issues.
# 
# These insights highlight general sentiment trends, user satisfaction levels, and specific areas for improvement. They can guide LinkedIn in making informed decisions to enhance the user experience and maintain a positive community.

# In[ ]:




