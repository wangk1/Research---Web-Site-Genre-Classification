from sklearn.feature_extraction import DictVectorizer

__author__ = 'Kevin'

import numpy as np
import scipy.sparse as sp

"""

"""

def load_train_matrix(*,train_dv=None,train_coll_cls,stack_per_sample=3000):

    train_bows=None
    train_labels=[]

    matrix_cache=[]
    for count,train_bow_obj in enumerate(train_coll_cls.objects):
        if count %1000==0:
            print("Train load curr at:  {}".format(count))

        curr_bow_matrix=train_dv.transform(train_bow_obj.attr_map)[0]
        matrix_cache.append(curr_bow_matrix)
        train_labels.append(train_bow_obj.short_genre)

        if len(matrix_cache)>stack_per_sample:
            train_bows=sp.vstack(matrix_cache)
            matrix_cache=[train_bows]
            print("stacked, train bow size:{},labels size: {}".format(train_bows.shape[0],len(train_labels)))



    if len(matrix_cache)>1:
        print("stacking")
        train_bows=sp.vstack(matrix_cache)
        matrix_cache=[]

    print("Final training size: {}".format(train_bows.shape[0]))
    return train_bows,np.asarray(train_labels)

def load_test_matrix(*,test_dv=None,test_coll_cls):

    test_bows=[]
    test_labels=[]
    bow_indexes=[]

    explored=set()
    for count,test_bow_obj in enumerate(test_coll_cls.objects):
        if count %1000==0:
            print("Test load curr at:  {}".format(count))

        if test_bow_obj.ref_index in explored:
            continue

        explored.add(test_bow_obj.ref_index)

        test_labels.append(test_bow_obj.short_genre)
        test_bows.append(test_bow_obj.attr_map)
        bow_indexes.append(test_bow_obj.ref_index)
    if test_dv is None:
        test_dv=DictVectorizer()
        return test_dv.fit_transform(test_bows),np.asarray(test_labels),np.asarray(bow_indexes)

    else:
        return test_dv.transform(test_bows),np.asarray(test_labels),np.asarray(bow_indexes)


