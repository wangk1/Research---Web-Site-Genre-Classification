import collections
import re
"""
This module is geared towards providing functions that transforms the classification result file.

Current functions:
    ClassificationInstance, namedtuple that represent an entry in the classficiation result file
    get_classification_res, function that transforms classfication result file to actual ClassificationInstance objs.

"""

__author__ = 'Kevin'

ClassificationInstance=collections.namedtuple("ClassificationInstance",("ref_id","actual","predicted","classifier"))
def get_classification_res(filepath):
    """
    Convert a result file into multiple objects

    :return: list of ClassificationInstance objects that represents an id of webpage and its actual, predicted, and classifier.
    """
    prediction_objs=[]

    classifier=re.search(".*(?=_wrong.txt)",filepath.split("\\")[-1]).group(0)
    with open(filepath) as sample_file:
        for line in sample_file:

            res_list=line.split(" ")

            prediction_objs.append(ClassificationInstance(ref_id=res_list[0][:-1],predicted=res_list[2][:-1],actual=res_list[4][:-1],
                                                    classifier=classifier))

    return prediction_objs