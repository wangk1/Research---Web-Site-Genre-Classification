__author__ = 'Kevin'

import numpy as np
import scipy.sparse as sp

import pickle,itertools

from sklearn.feature_extraction import DictVectorizer
from util.base_util import normalize_genre_string

from util.Logger import Logger

util_logger=Logger()
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


def pickle_obj(obj,file_path):
    """
    Pick the object at file at file_path. Will overwrite files so be careful

    :param file_path:
    :param obj:
    :return:
    """

    util_logger.info("Pickling")
    with open(file_path,mode="wb") as pickle_file_handle:
        pickle.dump(obj,pickle_file_handle)
    util_logger.info("Pickling done")

def unpickle_obj(file_path):
    """
    Unpickle the file stored at path of file_path

    :param file_path:
    :return:
    """

    util_logger.info("UnPickling")

    with open(file_path,mode="rb") as pickle_file_handle:
        obj=pickle.load(pickle_file_handle)


    return obj

def flatten_training(train_set):
    """
    Flatten the training set where there may be multiple genre classes. Split that into 2

    :param train_set:
    :return:
    """
    new_y=[]
    new_x_index=[]
    new_ref_id=[]

    for index,g_list in enumerate(train_set.y):
        new_ref_id.extend((len(g_list))*[train_set.ref_indexes[index]])
        new_x_index.extend([index]*len(g_list))
        new_y.extend(g_list)

    #now project out
    train_set.X=train_set.X[new_x_index]
    train_set.y=np.array(new_y)
    train_set.ref_indexes=np.array(new_ref_id)

def normalizer(y,level=1):
    """
    Utility function for automatically normalizing a vector of list or a vector of genres to level @param level.

    Note that this function is not in place, a new object is created
    :param y:
    :param level:
    :return:
    """
    if np.issubdtype(y.dtype,np.str):
        new_y=np.array([normalize_genre_string(i,level) for i in y])

    else:
        new_y=np.array([list(set(normalize_genre_string(g,level) for g in y_list))  for y_list in y])

    return new_y