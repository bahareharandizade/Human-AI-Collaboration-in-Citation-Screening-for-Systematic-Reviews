from nltk.stem.snowball import SnowballStemmer
import  nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup




class vectorize(object):

    #vectorize

    @staticmethod
    def bagofwords1(learning,resting, tokenizer=None, ngram_range_min=1,ngram_range_max=3, stop_words=None, lowercase=True, max_df=1.0,min_df=1, max_features=None):

        BOWvectorizer = CountVectorizer(tokenizer=None,
                                        analyzer="word",
                                        ngram_range=(ngram_range_min,ngram_range_max),
                                        stop_words=stop_words,
                                        lowercase=lowercase,
                                        max_df=max_df,
                                        min_df=min_df,
                                        max_features=max_features)

        learn = tokenizer(learning)
        rest=tokenizer(resting)

        BOWrest=BOWvectorizer.fit_transform(rest)
        BOWlearn=BOWvectorizer.transform(learn)

        return BOWlearn,BOWrest

    @staticmethod
    def tfidftransform1(learning,resting):

        tfidftransform = TfidfTransformer(norm="l2",
                                          use_idf=True,
                                          smooth_idf=True,
                                          sublinear_tf=False)

        learningtfidf=tfidftransform.fit_transform(resting)
        resttfidf=tfidftransform.transform(learning)


        return learningtfidf,resttfidf

    @staticmethod
    def TfidfVectorizer1(learning,resting, tokenizer=None, ngram_range_min=1,ngram_range_max=1, stop_words=None, lowercase=True, max_df=1.0,min_df=1, max_features=None):
        vector = TfidfVectorizer(tokenizer=None,
                                 analyzer="word",
                                 ngram_range=(ngram_range_min, ngram_range_max),
                                 stop_words=stop_words,
                                 lowercase=lowercase,
                                 max_df=max_df,
                                 min_df=min_df,
                                 max_features=max_features)
        learn=tokenizer(learning)
        rest=tokenizer(resting)

        tfidfrest=vector.fit_transform(rest)
        tfidflearn=vector.transform(learn)#the rest is transfrom to what we fit for train!

        return tfidflearn,tfidfrest


    #tokenize

    @staticmethod
    def tokenize_and_stem(text):
        cleansent = []
        stemmer = SnowballStemmer("english")
        for sent in text:
            sent = " ".join(sent)
            sent = BeautifulSoup(sent,"html.parser").get_text()
            text = re.sub("[^a-zA-Z]", " ", sent)
            tokens = text.lower().split()
            stems = [stemmer.stem(t) for t in tokens]
            cleansent.append(" ".join(stems))
        return cleansent

    @staticmethod
    def tokenize_only(text):



        cleansent=[]
        for sent in text:
            sent=" ".join(sent)
            sent = BeautifulSoup(sent,'html.parser').get_text()
            sent=re.sub("[^a-zA-Z]"," ", sent)
            tokens = sent.lower().split()
            cleansent.append(" ".join(tokens))
        return cleansent



    @staticmethod
    def review_to_sentences( text):

        cleansent = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for review in text:

            raw_sentences = tokenizer.tokenizer(review.decode('utf8').strip())


            sentences = []
            for raw_sentence in raw_sentences:

                if len(raw_sentence) > 0:

                    sentences.append(vectorize.tokenize_only(raw_sentence))
            cleansent+=sentences

        return cleansent



