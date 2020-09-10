# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Read: WIP

import GetOldTweets3 as got
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

import numpy as np
from PIL import Image
import datetime

"""--------------------------------------------------------------------
----------------- WordCloud Visual of Trump's Twitter -----------------
--------------------------------------------------------------------"""

#Username of Twitter Account to scrape and Count of Tweets
username = "@realDonaldTrump"
count = 10000

#Setting up the TweetCriteria
tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(count)
    
#List of Tweets with Timestamps and Tweet Text
tweets = got.manager.TweetManager.getTweets(tweetCriteria)

#Timestamped + Tweet, Tweet
text_tweets = [[x.date, x.text] for x in tweets]
puretext = [x.text for x in tweets]

#Generating Wordcloud of Last 10000 Tweets
puretextString = " ".join(puretext)
wordcloud = WordCloud().generate(puretextString)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Setting up Stop Words
stopwords = list(stopwords.words("english")) 
wordtokenize = word_tokenize(puretextString.lower())

stopwords.append("thank")
stopwords.append("you")
stopwords.append("That")
stopwords.append("We")

#Empty list for words
djTrumpNoFiller = []

#Filter out common words
for x in wordtokenize:
    if x not in stopwords:
        djTrumpNoFiller.append(x)
 
#Filtered word cloud
filteredPureTextString = " ".join(djTrumpNoFiller)
filteredWordcloud = WordCloud().generate(filteredPureTextString)

plt.imshow(filteredWordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#American Flag
mask = np.array(Image.open(r"AmericanFlag.png"))

americanFlagWordCloud = WordCloud(stopwords=stopwords, background_color="white", \
                          mode="RGBA", max_words=750, mask=mask, min_font_size = 2, \
                              max_font_size = 1000, min_word_length = 3)

americanFlagWordCloud.generate(filteredPureTextString)
    
#Coloring
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[10,10])
plt.imshow(americanFlagWordCloud.recolor(color_func=image_colors), \
           interpolation="bilinear")
plt.axis("off")

plt.show()

"""--------------------------------------------------------------------
----------------------- Covid 19 Twitter Analysis ---------------------
--------------------------------------------------------------------"""

#DataFrame of Timestamps and Tweets stored in Date, Tweet columns
tweet_df = pd.DataFrame(text_tweets, columns = ["Date", "Tweet"])


#Terms to search tweets
covid19TweetsCheckList = ["Covid", "Covid-19", "Covid19", "ChinaVirus", "Corona", \
                 "Coronavirus", "ChinaFlu", "KungFlu", "China Flu", "Kung Flu", 
                 "Virus", "Vaccine", "Vaccines", "Flu", "Disease", "Sickness", \
                     "Mask", "Masks"]

    
#DataFrame for storing Covid-Related Tweets
covid_df = pd.DataFrame(columns = ["Date", "Tweet"])

conditions = False
for x in covid19TweetsCheckList:
    conditions = conditions | (tweet_df.Tweet.str.contains(x, case = False))
covid_df = tweet_df.loc[conditions]

covid_df = covid_df.reset_index()
covid_df = covid_df.drop(columns = "index")

covid_df.to_csv("covidtweets.csv")

#Convert Date to Datetime
covid_df["Date"] = pd.to_datetime(covid_df["Date"], format = '%m/%d/%Y')
covid_df["Date"] = covid_df["Date"].dt.tz_localize(None)

#Creating new DataFrame of Covid Related Tweets After March
march_july_covid_df = pd.DataFrame(columns = ["Date", "Tweet"])

dummydate = datetime.datetime(2020, 3, 1)
conditions = False
for x in covid_df.Date:
    conditions = conditions | (covid_df.Date > dummydate)
march_july_covid_df = covid_df.loc[conditions]


march_july_covid_df["Time"] = march_july_covid_df["Date"].dt.strftime("%H")

#Frequency Distribution of Covid-Related Tweets By Hour of Day
hourOfDay_covid = nltk.FreqDist(march_july_covid_df["Time"])
hourOfDay_covid_df = pd.DataFrame({"Hour Of Day": list(hourOfDay_covid.keys()), \
                      'Count': list(hourOfDay_covid.values())})
    
HOD_top10 = hourOfDay_covid_df.nlargest(columns="Count", n = 24) 

plt.figure(figsize=(20,7))
HOD_bg = sns.barplot(data = HOD_top10, x= "Hour Of Day", y = "Count")
HOD_bg.set(ylabel = "Count")
plt.show()


#-----------------------------------------------------------------------------


""" Covid 19 Cases, Deaths by Jurisdiction; 
    Source: https://ourworldindata.org/coronavirus/country/united-states?country=~USA
    Date: 2020-09-08
    Includes only United States in country
    Filtered to exclude Jan, Feb """
    
covidPostMarch_df = pd.read_csv(r"owid-covid-data1.csv", \
                                skiprows=[x for x in range(1,62)])

covidPostMarch_df.rename(columns = {"date":"Date"}, inplace = True) 
covidPostMarch_df["Date"]= pd.to_datetime(covidPostMarch_df["Date"])

covidPostMarch_df.plot(x = "Date", y = "new_cases", linewidth = 1, fontsize = 7)


#-----------------------------------------------------------------------------


#Extracting Hashtags
hashtagsDJT = []
for x in puretext:
    hashtags = re.findall(r"#(\w+)", x)
    hashtagsDJT.append(hashtags)

hashtagsDJT

hashtagsDJT = sum(hashtagsDJT,[])


HTFreq = nltk.FreqDist(hashtagsDJT)
HT_df = pd.DataFrame({'Hashtag': list(HTFreq.keys()), \
                      'Count': list(HTFreq.values())})

    
#Most Popular Hashtags
HT_top10 = HT_df.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(20,7))
HT_bg = sns.barplot(data=HT_top10, x= "Hashtag", y = "Count")
HT_bg.set(ylabel = 'Count')
plt.show()


"""------------------------ Sentiment Analysis -------------------------------
---------------------------------------------------------------------------"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

SAtweets = text_tweets    
SAtweets_df = pd.DataFrame(SAtweets, columns = ["Date", "Tweet"])

SAtweets_list = puretext

while('' in SAtweets_list):
    SAtweets_list.remove('')

nan_value = float("NaN")
SAtweets_df.replace("", nan_value, inplace = True)
SAtweets_df.dropna(subset = ["Tweet"], inplace = True)
SAtweets_df = SAtweets_df.reset_index(drop=True)

tweet_scored = []

for x in SAtweets_list:
    score = analyser.polarity_scores(x)
    tweet_scored.append(score)
    
score_df = pd.DataFrame(tweet_scored)
score_df.mean()

scoredtweets_df = SAtweets_df.join(score_df)
scoredtweets_df.to_csv("scoredtweets_df.csv")

TS_sentiment = pd.DataFrame(scoredtweets_df, columns = ["Date", "compound"])

covid_Sent = pd.DataFrame(columns = ["Date", "compound"])
TS_sentiment["Date"] = TS_sentiment["Date"].dt.tz_localize(None)

dummydate = datetime.datetime(2020, 3, 1)
conditions = False
for x in TS_sentiment.Date:
    conditions = conditions | (TS_sentiment.Date > dummydate)
covid_Sent = TS_sentiment.loc[conditions]

#-----------------------------------------------------------------------------

covid_Sent["Date"] = covid_Sent["Date"].dt.date

means = covid_Sent.groupby("Date").mean()
means = means.reset_index()

fig, ax1 = plt.subplots(1,1)
covidPostMarch_df.plot(y = "new_cases", ax=ax1, linewidth = 1, fontsize = 7, \
                       color='blue', label='New Cases')
ax2 = ax1.twinx()
means.plot(y = "compound", ax=ax2, linewidth = 1, fontsize = 7, \
           color='green', label='Average Sentiment')
ax1.legend(loc=3)
ax2.legend(loc=4)
plt.show()
