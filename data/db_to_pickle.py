
from sklearn.feature_extraction import DictVectorizer

import collections as coll,os
import numpy as np
import settings
from data.util import pickle_obj

__author__ = 'Kevin'

"""
Module containing Python functions that transforms database data to pickle files in settings.PICKLE_DIR
"""

def db_to_pickle(src_db,secondary_path=""):
    """
    Convert database data to pickle file of X,y, and ref_index.

    The database object must have the following properties
        ref_index: callback int assignment to URLToGenre
        attr_map: dictionary of attribute to count
        short_genres: Genres for each attr_map, aka its label(s)

    Store in PICKLE_DIR/$secondary_path/
    :return: X,y,ref_index
        y is nx1 where n is number of labels. It is an np array of lists where list are all the genres an instance may
            have.
    """
    if secondary_path == "":
        print("No secondary path set.")

    vocabulary_set=set()
    for all_gram_obj in src_db.objects:
        vocabulary_set |=set(all_gram_obj.attr_map.keys())
        del all_gram_obj

    print("The size of the url vocabulary: {}".format(len(vocabulary_set)))
    vocabulary_dict=coll.Counter((i for i in vocabulary_set))

    print("Fitting vocabulary")
    vectorizer=DictVectorizer()
    vectorizer.fit([vocabulary_dict])

    print("Transforming")
    stack=500
    X_stack=[]
    y_stack=[]
    ref_index=[]
    for c,all_gram_obj in enumerate(src_db.objects.no_cache()):
        c%10000==0 and print(c)

        X_stack.append(all_gram_obj.attr_map)
        y_stack.append(all_gram_obj.short_genres)
        ref_index.append(all_gram_obj.ref_index)
        del all_gram_obj

    X=vectorizer.transform(X_stack)
    y=np.array(y_stack)
    ref_index=np.array(ref_index)

    #store x,y, and ref_index into pickle
    dir_path=os.path.join(settings.PICKLE_DIR,secondary_path)

    os.makedirs(dir_path,exist_ok=True)

    X_path=os.path.join(dir_path,"X_{}_pickle".format(secondary_path))
    y_path=os.path.join(dir_path,"y_{}_pickle".format(secondary_path))
    ref_path=os.path.join(dir_path,"refIndex_{}_pickle".format(secondary_path))
    vectorizer_path=os.path.join(dir_path,"vocab_vectorizer_{}_pickle".format(secondary_path))

    pickle_obj(X,X_path)
    pickle_obj(y,y_path)
    pickle_obj(ref_index,ref_path)
    pickle_obj(vectorizer,vectorizer_path)