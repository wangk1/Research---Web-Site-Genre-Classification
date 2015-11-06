__author__ = 'Kevin'

import nltk,re
from .word_based import NGrams

class URLTransformer:
    """
    This class takes url strings and transforms it based on the criterias present in this paper:
    http://infoscience.epfl.ch/record/136823/files/Topic_www_09.pdf
    into numpy feature array

    """

    def __init__(self):
        self.regex_tok=nltk.tokenize.RegexpTokenizer("[a-z]+")

    def transform(self,url):
        """
        Tokenize the url based on the rules:
        1. Lowercased
        2. http,https,www removed
        3. Split at punctuation, number or non-letter characters
        4. Eliminate tokens of length 2 or less

        Then, make all grams of the entire vocab and return the all gram object

        :param url: A url
        :return: FeatureUnion object
        """

        tokens=self.preprocess(url)

        return NGrams(analyzer="char").all_grams(*tokens)



    def preprocess(self,url):

        """
        Preprocess and returns tokens that are:
        1. Lowercased
        2. http,https,www removed
        3. Split at punctuation, number or non-letter characters
        4. Eliminate tokens of length 2 or less

        :param url:
        :return:
        """
        assert isinstance(url,str)

        #lowercase, remove www,https,http
        url=re.sub("http[s]?|www","",url.lower())

        # split on anything not a letter, eliminate tokens that are less or same as 2 in length
        return filter(lambda tok: len(tok)>2,self.regex_tok.tokenize(url))