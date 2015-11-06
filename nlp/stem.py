__author__ = 'Kevin'
from nltk import PorterStemmer


class EnglishStemmer:
    """
    Stemmer wrapper on Sumy's Stemmer for compatibility reasons with summarizer, but uses nltk's porter stemmer to do
    the actual stemming.

    """

    def __init__(self):
        self.__stemmer=PorterStemmer()

    def __call__(self, word):
        return self.__stemmer.stem(word)