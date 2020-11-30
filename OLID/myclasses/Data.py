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

from myclasses.Features import Features

class Data:
    def read(self, subtask="a", ids=None):
        df_tr = pd.DataFrame(pd.read_csv("olid-training-v1.0.tsv", sep="\t"))
            
        if subtask == "b" or subtask == "c":
            df_tr = df_tr[(df_tr[f"subtask_{subtask}"].notnull())]
            df_tr.reset_index(inplace=True)
        
        df_tr['tweet_original'] = df_tr['tweet']
        df_tr['tweet'] = self.preprocess_text_caller(df_tr['tweet'])
        
        df_te = pd.DataFrame(pd.read_csv(f"testset-level{subtask}.tsv",sep="\t"))
        df_te['tweet_original'] = df_te['tweet']
        df_te['tweet'] = self.preprocess_text_caller(df_te['tweet'])
        
        self.f = Features(df_tr, df_te)

        X_tr = self.feature_extraction(df_tr, self.f)
        ngram_ft = self.f.n_grams(df_tr["tweet"])
        X_tr = np.column_stack((X_tr, ngram_ft.toarray()))
        y_tr = df_tr[f"subtask_{subtask}"]
        
        y_te = pd.DataFrame(pd.read_csv(f"labels-level{subtask}.csv", header=None))
          
        if ids is not None:
            df_te = df_te[df_te.id.isin(ids)]
            df_te.reset_index(inplace=True)
            y_te = y_te[y_te[0].isin(ids)]
                
        X_te = self.feature_extraction(df_te, self.f)
        ngram_ft = self.f.n_grams(df_te["tweet"])
        X_te = np.column_stack((X_te, ngram_ft.toarray()))
                 
        self.df_tr = df_tr
        self.df_te = df_te
        
        return X_tr, X_te, y_tr, y_te[1]
    
    def preprocess_text(self, tweet, flag_stemm=True, flag_lemm=False):

        ## clean (convert to lowercase and remove punctuations and characters and then strip)
        tweet = re.sub(r'[^\w\s]', '', str(tweet).lower().strip())
        ## Tokenize (convert from string to list) and remove the stop words

        tokenize_tweet = tweet.split()

        ## Stemming (remove -ing, -ly, ...)
        if flag_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            tokenize_tweet = [ps.stem(word) for word in tokenize_tweet]

        ## Lemmatisation (convert the word into root word)
        if flag_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            tokenize_tweet = [lem.lemmatize(word) for word in tokenize_tweet]
    #     print(tokenize_tweet)
        ## back to string from list
        
        tweet = " ".join(tokenize_tweet)
        return tweet    
    
    def feature_extraction(self, df, f):
        Tweets_List = df["tweet_original"]
        
        df_feature = pd.DataFrame(f.FuncAllCapsCount(Tweets_List)).astype('int64')
        df_feature.rename(columns={0:"all_caps_count"},inplace=True)

        #Counting The Punctuations
        ContPunctuationCount, LastTokenPunctuation = f.FuncPunctuationsCount(Tweets_List)
        df_feature["Cont_Punc_Count"] = pd.DataFrame(ContPunctuationCount).astype('int64')
        df_feature["Last_Token_Punc"] = pd.DataFrame(LastTokenPunctuation).astype('int64')

        #Count the hash tags
        df_feature["HashTag_Count"] = pd.DataFrame(f.FuncHashtagsCount(Tweets_List)).astype('int64')
        df_feature["Anger word count"] = pd.DataFrame(f.Func_NRC_word_emotion_lexicon(Tweets_List)).astype('int64')
        df_feature["NRC_10_Expanded"] = pd.DataFrame(f.Func_NRC_10_Expanded_Emotion(Tweets_List))
        df_feature["Vader_Sentiment"]=pd.DataFrame(f.vader(Tweets_List))
        
        ReadingEase_list,ReadingGrade_List=f.reading_ease(Tweets_List)
        df_feature["Reading_Grade"]=pd.DataFrame(ReadingGrade_List)
        return(df_feature)
    
    def preprocess_text_caller(self, Tweets_List):
        Tweets_List_Processed = []
        
        for i in (range(len(Tweets_List))):
            Tweets_List_Processed.append(self.preprocess_text(Tweets_List[i]))
            
        return Tweets_List_Processed
    
    def ui_helper(self, tweets):
        df = pd.DataFrame(tweets)
        df.rename(columns={0:"tweets"},inplace=True)
        df["tweet_original"] = df["tweets"]
        df["tweets"] = self.preprocess_text_caller(df["tweets"])
        X = self.feature_extraction(df, self.f)
        ngram_ft = self.f.n_grams(df["tweets"])
        X = np.column_stack((X, ngram_ft.toarray()))
        return X