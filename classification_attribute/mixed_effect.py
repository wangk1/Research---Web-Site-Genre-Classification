import itertools

__author__ = 'Kevin'

from data.util import unpickle_obj,flatten_set,random_pick_samples,genre_normalizer,pickle_obj
from sklearn.feature_selection import chi2,SelectKBest
from util.Logger import Logger
import numpy as np
from db.db_model.mongo_websites_models import URLBow
from scipy.sparse import csc_matrix,lil_matrix,coo_matrix,vstack

mixed_effect_logger=Logger(__name__)

"""
Module to generate mixed effect X matrices.

"""




def _generate_mixed_effect_matrix(X_path,y_path,feat_selector):
    """
    Converts X to a COO Matrix of Mixed effect matrix

    :param X_path:
    :param y_path:
    :param feat_selector:
    :return:
    """

    mixed_effect_logger.debug("Flattening")

    #Reduce the column count
    X,y,_=flatten_set(*random_pick_samples(unpickle_obj(X_path),genre_normalizer(unpickle_obj(y_path))))
    feat_selector.fit(X,y)

    mixed_effect_logger.debug("Final size of X: {} y:{}".format(X.shape,y.shape))

    #Get the column selector, indices
    vocab_selector=feat_selector.get_support(True)
    num_vocab=vocab_selector.shape[0]

    vstack_list=[0]*X.shape[0]
    for ind,X_row in enumerate(X):
        ind % 10==0 and mixed_effect_logger.info("Done with {}".format(ind))

        row=np.zeros((1,num_vocab**2))
        select_col=X_row[0,vocab_selector].toarray() #convert to dense rep.

        #Compare each index to each row. Record the minimum as cooccurence
        for col_ind in range(0,select_col.shape[1]):
            if not select_col[0,col_ind]:
                continue

            cmp=np.full((1,select_col.shape[1]),fill_value=select_col[0,col_ind])
            select_col=np.minimum(select_col,cmp)
            row[0,col_ind*num_vocab:(col_ind+1)*num_vocab]=select_col

        vstack_list[ind]=lil_matrix(row)
        del row,select_col

    return vstack(vstack_list).tocoo()

def mixed_effect_chisq():
    k=5000
    X_path="../pickle_dir\\summary\\X_summary_pickle"
    y_path="../pickle_dir\\summary\\y_summary_pickle"
    vocab_vect_path="../pickle_dir\\summary\\vocab_vectorizer_summary_pickle"
    pickle_x_mixed_path="../pickle_dir/summary/X_mixed_summary_pickle"

    feat_selector=SelectKBest(chi2,k)

    X_mixed=_generate_mixed_effect_matrix(X_path,y_path,feat_selector)

    pickle_obj(X_mixed,pickle_x_mixed_path)

if __name__=="__main__":
    mixed_effect_chisq()