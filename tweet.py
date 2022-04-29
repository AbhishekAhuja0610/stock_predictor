import tweepy
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def gettweetdata(company):
    # My Twitter API Authentication VariablesS
    consumer_key = 'V7rQ8FI7vLKUjx5ILP7QvCF8i'
    consumer_secret = 'QF3VkzcRevJOU84jh6hdBg2JmCvsHxS58lDGKN4DTqVQshoD2s'
    access_token = '1319274840373620741-I25yTi0xnW3KaJEnjHmvspVtlcozCi'
    access_token_secret = 'TcynlymCwycoI3H9fyJXkrNpAg5ahvNudIjfYYUyX4dN2'

    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
 
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    tweets = api.search_tweets(company, count=1000)

    data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

    # print(data.head(50))

    import nltk
    nltk.download('vader_lexicon')
    # print(tweets[0].created_at)

    sid = SentimentIntensityAnalyzer()

    listy = []

    for index, row in data.iterrows():
        ss = sid.polarity_scores(row["Tweets"])
        listy.append(ss)

    se = pd.Series(listy)
    data['polarity'] = se.values

    data.to_csv(r'./polarity.csv')
    # print(listy)
    sums = 0.0
    for i in range(1, len(listy)):
        sums = sums + float(listy[i].get('compound'))
    return sums / len(listy)


