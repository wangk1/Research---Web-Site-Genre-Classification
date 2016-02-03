import itertools

__author__ = 'Kevin'

from data.util import unpickle_obj,flatten_set,random_pick_samples,genre_normalizer
from sklearn.feature_selection import chi2,SelectKBest
from util.Logger import Logger
import numpy as np
from db.db_model.mongo_websites_models import URLBow
from scipy.sparse import coo_matrix

mixed_effect_logger=Logger(__name__)

"""
Module to generate mixed effect X matrices.

"""


def _mixed_effect_dict(vocab_vect,fitted_feature_selector):
    """
    Generate mapping of vocab -> index so we can map vocab pairs -> another index in the sparse X_mixed array

    :param vocab_vect:
    :param fitted_feature_selector:
    :return:
    """
    vocab_selector=np.array(fitted_feature_selector.get_support())

    #convert to object array because np does not like long strings
    selected_vocab=np.array(vocab_vect.feature_names_,dtype=np.object)[vocab_selector].astype(np.str)

    return {c:v for c,v in enumerate(selected_vocab)}


def _generate_mixed_effect_matrix(X_path,y_path,vocab_vect_path,feat_selector,src_db_cls):
    mixed_effect_logger.debug("Flattening")

    X,y,_=flatten_set(*random_pick_samples(unpickle_obj(X_path),genre_normalizer(unpickle_obj(y_path))))
    feat_selector.fit(X,y)

    mixed_effect_logger.debug("Final size of X: {} y:{}".format(X.shape,y.shape))

    mixed_effect_dict=_mixed_effect_dict(unpickle_obj(vocab_vect_path),feat_selector)

    X_mixed=coo_matrix((src_db_cls.objects.count(),len(mixed_effect_dict)**2))
    ref_indexes=np.zeros((src_db_cls.objects.count()),dtype=np.int)
    for ind,db_obj in enumerate(src_db_cls.objects.no_cache()):
        ind % 10000==0 and mixed_effect_logger.info("Done with {}".format(ind))
        attr_map=db_obj.attr_map
        ref_indexes[ind]=db_obj.ref_index

        #identify words in the mixed effect dict
        word_list_present=[w for w in attr_map.keys() if w in mixed_effect_dict]

        for w1,w2 in itertools.product(word_list_present,word_list_present):
            #index in array=index_w1*len(mixed_effect_dict)+index_w2, where index is index is word_list_present[w1|w2]
            coo_index=word_list_present[w1]*len(mixed_effect_dict)+word_list_present[w2]

            #cooccurence of w1,w2 is the minimum of the count b/w those two
            X_mixed[ind,coo_index]=min(attr_map[w1],attr_map[w2])

    return X_mixed

def mixed_effect_chisq():
    k=30000
    X_path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir\\summary\\X_summary_pickle"
    y_path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir\\summary\\y_summary_pickle"
    vocab_vect_path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir\\summary\\vocab_vectorizer_summary_pickle"

    feat_selector=SelectKBest(chi2,k)

    X_mixed=_generate_mixed_effect_matrix(X_path,y_path,vocab_vect_path,feat_selector,URLBow)

if __name__=="__main__":
    mixed_effect_chisq()