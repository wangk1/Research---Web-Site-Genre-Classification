__author__ = 'Kevin'
import numpy as np

def filter_genres(X,y,ref_indexes,ignore_genre):
    """
    Cleans up the label set, remove those in the ignore list from the websites.

    :param X:
    :param y:
    :param ref_indexes:
    :return:
    """
    removal_count={g:0 for g in ignore_genre}

    for index,g_list in enumerate(y):
        y[index]=list(set(g_list)-ignore_genre)

    keep_index=np.array([i!=[] for i in y])

    print("Eliminated {} webpages".format(y.shape[0]-sum(keep_index)))

    return X[keep_index],y[keep_index],ref_indexes[keep_index]