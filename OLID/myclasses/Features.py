import pandas as pd
import nltk
import string
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import pickle
import emoji

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade

class Features:                    
    def __init__(self, df_tr, df_te):
        self.vectorizer_TFIDF = TfidfVectorizer(ngram_range = (1, 3), min_df=5, norm='l2')
        
        all_tweets = df_tr.tweet
        all_tweets=all_tweets.append(df_te.tweet)

        # fit was used to learn the vocabulary
        
        TempVectorTfidf = self.vectorizer_TFIDF.fit(all_tweets)
        
        # Getting the vocabulary for ngrams
        self.Vocab_ngrams_Tfidf = TempVectorTfidf.get_feature_names()

    def FuncAllCapsCount(self, GivenTweets):
        # Function to get All caps count from the tweets
        # pass tweets as it is
        
        AllCapsCount = np.zeros(len(GivenTweets))
        for i in tqdm(range(len(GivenTweets))):
            tweet = nltk.word_tokenize(GivenTweets[i])
            for word in tweet:
                if(word!="I" and word != "USER" and word != "URL" and re.match("^[A-Z]+$",word)):
                    AllCapsCount[i] += 1
                    
        return AllCapsCount
    
    def FuncPunctuationsCount(self, GivenTweets):
        # Function to get COntinuous Punctuation (???,??!,!!) Count and check if last token was a ? or !
        # pass tweets as it is
        
        ContPunctuationCount=np.zeros(len(GivenTweets))
        LastTokenPunctuation=np.zeros(len(GivenTweets))
        
        for i in tqdm(range(len(GivenTweets))):
            tweet = nltk.word_tokenize(GivenTweets[i])
            token = 0

            while token < len(tweet):
                if(tweet[token]=="?" or tweet[token]=="!"):
                    index=token+1
                    while(index< len(tweet) and (tweet[index]=="?" or tweet[index]=="!")):
                        index+=1
                    if(index-token>1):
                        ContPunctuationCount[i]+=1
                        token=index
                        
                token += 1
                
            if tweet[len(tweet)-1] == "?" or tweet[len(tweet)-1] == "!":
                LastTokenPunctuation[i] += 1

        return ContPunctuationCount, LastTokenPunctuation
    
    def FuncHashtagsCount(self, GivenTweets):    
        # Fucntion to get HashTag count from the tweets
        # pass tweets as it is
        
        HashTagsCount=np.zeros(len(GivenTweets))
        for i in tqdm(range(len(GivenTweets))):
            tweet=GivenTweets[i].split(" ")
            for word in tweet:
                if(re.match("^#[a-zA-Z][a-zA-Z0-9]*",word)):
                    HashTagsCount[i]+=1
        return(HashTagsCount)    
    
    def Func_NRC_Hashtag_Emotion(self, GivenTweets):
        # NRC-Hashtag-Emotion-Lexicon
        
        d_emotion = pd.read_csv("./lexicons/5. NRC-Hashtag-Emotion-Lexicon-v0.2.txt", sep='\t', names=['emotion','word','score'], header=None)
        dic_dEmo = {}
        for i in range(len(d_emotion['word'])):
            if d_emotion['emotion'][i]=="anger":
                dic_dEmo[d_emotion['word'][i]]=d_emotion['score'][i]
        aggEmoHashtags=np.zeros(len(GivenTweets))
        for i in range(len(GivenTweets)):
            tweet=GivenTweets[i]
            hashtags = [j  for j in tweet.split() if j.startswith("#") ]
            for tag in hashtags:
                word=tag.lower()
                if word in dic_dEmo:
                    aggEmoHashtags[i]+=dic_dEmo[word]
                elif word[1:] in dic_dEmo:
                    aggEmoHashtags[i]+=dic_dEmo[word[1:]]
                    
        return(aggEmoHashtags)

        #     Vector aggEmoHashtags conatains Aggregate emotion score (Hashtags)
        #     Adding features into dataframe
        
    def Func_NRC_10_Expanded_Emotion(self, GivenTweets):
        # NRC-Hashtag-Emotion-Lexicon
        
        d_emotion=pd.read_csv("./lexicons/6. NRC-10-expanded.csv", sep='\t')
        dic_dEmo={}
        for i in range(len(d_emotion['word'])):
            dic_dEmo[d_emotion['word'][i]]=d_emotion['anger'][i]
        aggScoreEmo=np.zeros(len(GivenTweets))
        for i in range(len(GivenTweets)):
    #         print(i)
            tweet=GivenTweets[i]
            for word in word_tokenize(tweet):
                if word in dic_dEmo:
    #                 print(word)
                    aggScoreEmo[i]+=dic_dEmo[word]
        return(aggScoreEmo)      

    def Func_NRC_word_emotion_lexicon(self, GivenTweets):
        emotion_count=np.zeros(len(GivenTweets))
        emotions_df=pd.read_csv("./lexicons/8. NRC-word-emotion-lexicon.txt", sep='\t', names=['word','emotion','score'], header=None)
        # This set contains words with Emotion X=1 in above dataframe
        set_emo=set()
        for i in range(len(emotions_df['word'])):
            if emotions_df['emotion'][i] == "anger" and emotions_df['score'][i]==1:
                set_emo.add(emotions_df['word'][i])

        for i in range(len(GivenTweets)):
            tweet=word_tokenize(GivenTweets[i])
            for word in tweet:
                word_lower=word.lower()
                if word_lower in set_emo:
                    emotion_count[i] += 1
                    
        return(emotion_count)    
        
    def n_grams(self, tweets, flag="tfidf"):
        if(flag=="tfidf"):
            ngrams_ft = self.vectorizer_TFIDF.transform(tweets)

        return ngrams_ft

    def slang(self, GivenTweets):
        slangs= pd.read_csv("./lexicons/SlangSD/SlangSD.txt", sep='\t', names=['word','score'], header=None)
        slangScore=np.zeros(len(GivenTweets))
        dic_slangs={}
        
        for i in (range(len(slangs['word']))):
            dic_slangs[slangs['word'][i]]=slangs['score'][i]
            
        for i in (range(len(GivenTweets))):
            tweet=word_tokenize(GivenTweets[i])
            for j in range(len(tweet)):
                if(tweet[j] in dic_slangs):
                    slangScore[i] += dic_slangs[tweet[j]]
                    
        return(slangScore)
    
    def vader(self, GivenTweets):
        obj = SentimentIntensityAnalyzer()
        Comp_vader = np.zeros(len(GivenTweets))
        for i in range(len(GivenTweets)):
            sentiment_dict = obj.polarity_scores(GivenTweets[i])
            Comp_vader[i]=sentiment_dict['compound']
        return(Comp_vader)
        
    def reading_ease(self,GivenTweets):
        Reading_Ease=np.zeros(len(GivenTweets))
        Reading_Grade=np.zeros(len(GivenTweets))
        for i in range(len(GivenTweets)):
            Reading_Ease[i]=flesch_reading_ease(GivenTweets[i])
            Reading_Grade[i]=flesch_kincaid_grade(GivenTweets[i])
        return(Reading_Ease,Reading_Grade)