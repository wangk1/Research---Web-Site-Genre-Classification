import collections
import itertools
import operator
from mongoengine.base import BaseDict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from nlp.stem import EnglishStemmer

__author__ = 'Kevin'


class BagOfWords:
    """
    Basic Bagof words model implemented with SciKit-Learn's sparse counting vectorizers
    """
    def __init__(self,**kwargs):
        self.vectorizer_args=kwargs
        self.vectorizer=CountVectorizer(decode_error='ignore',**self.vectorizer_args)

    def __call__(self,*txt,**kwargs):
        """
        Use SciKit vectorizer to transform the txt into an matrices of numbers represeting pure bag of words
        :return:
        """
        return self.vectorizer.fit_transform([str(i) for i in txt],**kwargs)

    def get_word_count(self, *txt,**kwargs):
        """
        First, fit to vocab and get word count

        Count occurence of each word in the bag of words representation from txt list, *txt

        Returns UNSORTED LIST
        """
        self.vectorizer.fit(txt)
        analyze = self.vectorizer.build_analyzer()


        return TextUtil.stem(
            TextUtil.remove_stop_words(collections.Counter(itertools.chain(*(analyze(str(i)) for i in txt))))
        )



    def get_feature_names(self):
        """
        Get the feature names of the vectorized vector

        :return:
        """
        return self.vectorizer.get_feature_names()

    def reverse_transformation(self,bow_dict):
        """
        Reverse the transformation of a dictionary representation of BOW into numpy vectors

        :return:
        """
        assert isinstance(bow_dict,BaseDict) or isinstance(bow_dict,dict)

        vec=DictVectorizer()
        vec.fit_transform(bow_dict)

        return vec


class NGrams(BagOfWords):
    """
    NGram vectorizer, generic

    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def all_grams(self,*txt):
        """
        Makes 4,5,6,8 grams of the input text

        :returns CountVector result of the aforementioned grams
        """

        self.set_grams(ngram_range=(3,8))
        analyze=self.vectorizer.build_analyzer()

        return collections.Counter(itertools.chain(*(analyze(i) for i in txt)))

    def set_grams(self,ngram_range=(1,1)):
        self.vectorizer_args['ngram_range']=ngram_range
        self.vectorizer=CountVectorizer(decode_error='ignore',**self.vectorizer_args)

class TextUtil:
    """
    Base text feature utils
    """


    stemmer=EnglishStemmer()

    @staticmethod
    def stem(word_dict):
        """
        Stems and combines word count with the same stem

        :param word_dict:
        :return:
        """

        new_dict={}
        for w in word_dict.keys():
            stemmed_w=TextUtil.stemmer(w)

            new_dict[stemmed_w]=(new_dict.get(stemmed_w,0) if stemmed_w!=w else 0)+word_dict[w]

        return new_dict

    @staticmethod
    def remove_stop_words(word_dict,lang="english",additional=set()):
        """
        Removes stop words according to a language, default is english, and in addition a set of additional
        user provided stop words.

        <i>Note that a new dictionary is generated</i>

        :param word_dict: A bow dictionary of feature and its respective count
        :param lang:
        :param additional:
        :return:
        """
        stop_words=set(stopwords.words(lang))|additional

        return dict((k,v) for k,v in word_dict.items() if k not in stop_words)
