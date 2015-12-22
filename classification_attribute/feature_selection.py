import itertools

__author__ = 'Kevin'
from sklearn.feature_selection import SelectKBest,chi2 as chi_sq
from scipy.sparse.linalg import svds
from scipy.sparse import identity

from util.text_preprocessor import preprocess
from .word_based import BagOfWords
from data.util import load_train_matrix
from util.Logger import Logger
from util.base_util import normalize_genre_string

feature_logger=Logger()

def feature_selection(train_set,test_set,feature_selector,fit=True):
        """
        Perform feature selection, applied to the whole dataset. Must be done before loading testing sets

        :param feat_selector: The feature selector
        :return:
        """

        assert hasattr(feature_selector,"transform")

        feature_logger.info("Pre feature selection: num features: {}".format(train_set.X.shape[1]))

        if fit:
            feature_selector.fit(train_set.X,train_set.y)

        train_set.X=feature_selector.transform(train_set.X)

        feature_logger.info("Post feature selection Train set: num features: {}".format(train_set.X.shape[1]))

        test_set.X=feature_selector.transform(test_set.X)
        feature_logger.info("Post feature selection Test set: num features: {}".format(test_set.X.shape[1]))

class SingleClassFeatureSelector:
    """


    """
    def __init__(self,transformers):

        self.transform=transformers


    def transform(self,X,y):


        #Get all the classes first
        genre_set=set(itertools.chain(*([normalize_genre_string(g,1) for g in g_list] for g_list in y)))

        for g in genre_set:
            genre_matches=[g in g_list for g_list in y]

        pass
class SparseSVD:

    def __init__(self,k):
        self.k=k

    def transform(self,X):
        lu,s,ru=svds(X,k=self.k)
        s=identity(self.k,dtype='Float64',format='dia').multiply(s)

        return lu.multiply(s).multiply(ru)



def chi_squared_feature_select(X,y,k_best=1500):
    """
    Rank the best attributes for discrimination based on chi square test.

    Puts the best attributes in the ChiSquared collection.
    :param X: Training sample
    :param y: Training labels
    :return: None
    """

    #load training

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

