__author__ = 'Kevin'
from data.util import unpickle_obj
import numpy as np


def load_X_y(path_X,path_y):
    return unpickle_obj(path_X), unpickle_obj(path_y)

def match_sets_based_on_ref_id(Xs,ys,ref_ids):
    """
    Orient the list of training set, labels, and ref_ids based on ref id so that each index j of X[i][j], y[i][j],
    ref_ids[i][j] corresponds to the same label and ref ids.

    :param Xs:
    :param ys:
    :param ref_ids:
    :return:
    """
    #greatest common denominator
    smallest_index=min(range(0,len(ref_ids)),key=lambda x:len(ref_ids[x]))
    ref_ids_to_index_loc={ref_id:ind for ind,ref_id in enumerate(ref_ids[smallest_index])}

    if len(Xs)>1:
        for index in range(0,len(ref_ids)):
            if index==smallest_index:
                continue
            c_ref_ids=ref_ids[index]
            c_X=Xs[index]
            c_y=ys[index]

            #project out those ref ids in the smallest array of ref ids
            selector=np.vectorize(lambda x:x in ref_ids_to_index_loc)(c_ref_ids)
            print("{} ids have been selected".format(sum(selector)))

            c_ref_ids=c_ref_ids[selector]
            c_X=c_X[selector]
            c_y=c_y[selector]

            #now remap each index
            remapped_index=np.vectorize(lambda x:ref_ids_to_index_loc[x])(c_ref_ids)

            Xs[index]=c_X[remapped_index]
            ys[index]=c_y[remapped_index]
            ref_ids[index]=c_ref_ids[remapped_index]

    return Xs,ys,ref_ids