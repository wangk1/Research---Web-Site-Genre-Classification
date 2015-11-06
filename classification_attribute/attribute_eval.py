__author__ = 'Kevin'
import numpy as np
from classification_attribute.word_based import BagOfWords

def tree_eval(X,y):
    """
    Evaluate X's fit on y and get the good attributes

    :param X:
    :param y:
    :return:
    """
    pass

def mutual_info():
    pass

def evaluate_attributes(bow_dicts,y):
    """
    Pass in multiple bow dictionary of features X and their corresponding labels: y.

    :param bow_dicts:
    :param y:
    :return:
    """
    #array the list
    y=np.array(y)
    bow_model=BagOfWords()

    X=[]

    #get the array of all the bows
    for bow_dict in bow_dicts:
        X.append(bow_model.reverse_transformation(bow_dict))

    exit()

