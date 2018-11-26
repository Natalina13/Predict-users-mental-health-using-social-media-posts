import codecs, json
import pandas as pd
from textblob import TextBlob
import re
import numpy as np

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def textbl(tweet):
	text=clean_tweet(tweet)
	analysis = TextBlob(text) 
	return analysis.sentiment.polarity

with codecs.open('C:/Users/lina9/Downloads/grad_studies/adbms/predict for depression/tweets.json', 'r', 'utf-8') as f:
    tweets = json.load(f, encoding='utf-8')

list_tweets = [list(elem.values()) for elem in tweets]
list_columns = list(tweets[0].keys())
df = pd.DataFrame(list_tweets, columns=list_columns)

df['sent']=np.array([ str(textbl(tweet)) for tweet in df['text'] ])
print(df['text']+','+df['sent'], end='')
