__author__ = 'Kevin'
from sklearn.feature_selection import SelectKBest,chi2 as chi_sq

from util.text_preprocessor import preprocess
from .word_based import BagOfWords
from classification.util import load_train_matrix,load_test_matrix



def chi_squared_feature_select(k_best=1500):
    """
    Rank the best attributes for discrimination based on chi square test.

    Puts the best attributes in the ChiSquared collection.
    :param X: Training sample
    :param y: Training labels
    :return: None
    """

    #load training
    X,y=load_train_matrix()

    return SelectKBest(chi_sq,k_best).fit_transform(X,y)


def convert_to_bow_dict(html_page):
    """
    Convert html page to plain text page and then to bow model without stop words and with porter stemming

    :param html_page: the html pages
    :return:
    """
    text_page=preprocess(html_page)
    bow_model=BagOfWords()

    return bow_model.get_word_count(text_page)

