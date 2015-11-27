import collections
import os
import re
import pandas as pd
import ast
import settings
import util.Logger as Logger

from db.db_model.mongo_websites_models import URLBow
from util.base_util import normalize_genre_string
from classification.util import unpickle_obj,pickle_obj

classification_res_logger=Logger.Logger()


"""
This module is geared towards providing functions that transforms the classification result file.

Current functions:
    ClassificationInstance, namedtuple that represent an entry in the classficiation result file
    get_classification_res, function that transforms classfication result file to actual ClassificationInstance objs.

"""

__author__ = 'Kevin'


CLASSIFICATION_HEADERS=["ref id","Predicted","Actual"]

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

            prediction_objs.append(ClassificationResultInstance(ref_id=res_list[0][:-1],predicted=res_list[2][:-1],actual=res_list[4][:-1],
                                                    classifier=classifier))

    return prediction_objs

def create_true_results(res_path):
    """
    Create files of actual misses,swings, and rights

    :param res_path:
    :return:
    """
    pass

class ClassificationResultInstance:

    def __init__(self,ref_id,actual,predicted,classifier,genre_lv=1):
        self.ref_id=ref_id
        self.__actual=actual
        self.predicted=predicted
        self.classifier=classifier
        self.genre_lv=genre_lv

    def __str__(self):
        return "{}, predicted: {}, actual: {}".format(self.ref_id,self.predicted,self.actual)

    def __repr__(self):
        return self.__str__()

    @property
    def actual(self):
        """
        Get all the genres of the instance. Normalized to the level of self.genre_lv

        :return:
        """
        return [normalize_genre_string(g,self.genre_lv) for g in self.__actual]


    def is_swing_sample(self,top_x_predicted=1):
        """
        Test if the ClassificationResultInstance object is a swing instance, its predicted class is within one of its
        multiple classes. So, right predictions are automatically also swing instances. But, wrong predicted samples
        may be a swing instance

        :param: top_x_predicted: check if the top x predictions are in the class's genres. If they all are,
        :return: True or False if the sample is swing instance
        """

        #grab all short genres and see if it matches
        url_bow_obj=URLBow.objects(index=self.ref_id).only("short_genres")[0]

        return all(pred_g in (normalize_genre_string(g,self.genre_lv) for g in url_bow_obj.short_genres) for pred_g in self.predicted[:top_x_predicted])

class ClassificationResultStream:
    """
    Base class for streaming prediction results from the text files, regardless of the results
    being wrongly predicted for correctly predicted

    Most of the time, the subclasses RightResultsStreamIter or WrongResultsStreamIter should be used
    """

    def __init__(self,result_path,classifier=None,get_all_actual=True):
        """
        Read results from the path of the results and prepare the iterator.

        The files in the result folder should follow the classification result file interface

        :param result_path: The path to folder containing all the classifier results
        :param classifier: Allow use to only stream the data from the results of one classifier.
            Can be an iterable or just a string along
        """

        self.result_path=result_path
        self.get_all_actual=get_all_actual

        #set of classifiers to look for
        self.classifiers=classifier if not isinstance(classifier,str) else (classifier,)


    def get_classification_res(self,filepath):
        """
        Convert a result file into the ClassificationResultInstance objects

        :return: list of ClassificationInstance objects that represents an id of webpage and its actual, predicted, and classifier.
        """
        prediction_objs=[]

        classifier=re.search(".*(?=_(wrong|right).txt)",filepath.split("\\")[-1]).group(0)

        file_datagram=pd.read_csv(filepath)

        for f in file_datagram.values:
            actual=[f[2]]

            #if self.get_all_actual:
                #actual=URLBow.objects.get(index=f[0]).short_genres

            prediction_objs.append(ClassificationResultInstance(ref_id=f[0],predicted=ast.literal_eval(f[1])
                                                                ,actual=actual,
                                                        classifier=classifier))

        return prediction_objs

    def get_classification_res_gen(self,suffix=".txt"):
        """
        Generator that generates the next classification result entry.

        It will read files with only specified suffix, so this can be used to filter
            out the right files or the wrong files or vice versa or any files really.

        :return:
        """
        abs_result_path=os.path.abspath(self.result_path)

        #in case we only chose some classifiers
        if self.classifiers is not None:
            right_files=(f for f in os.listdir(abs_result_path)
                         if f.endswith(suffix) and any(f.startswith(c) for c in self.classifiers))

        else:
            right_files=(f for f in os.listdir(abs_result_path) if f.endswith(suffix))

        for right_file in right_files:
            for right_instance in self.get_classification_res(os.path.join(abs_result_path,right_file)):
                yield right_instance

        raise StopIteration("End of right result stream")



class RightResultsIter(ClassificationResultStream):
    """
    Converts all the text files in a classification result directory that contains
        RIGHT/correct classification result into ClassificationResultInstance.
        Ignores any files that contains wrong results, aka the text files ending with _wrong.txt

    """

    def __init__(self,result_path,classifier=None):
        super().__init__(result_path,classifier)

        self.res_gen=super().get_classification_res_gen("_right.txt")

    def __next__(self):
        return next(self.res_gen)

    def __iter__(self):
        return self

    @staticmethod
    def load_iter_from_file(res_path,classifier=None,pickle_path=settings.PICKLE_DIR):
        right_iter_path=os.path.join(pickle_path,"right_res_iter")

        try:
            right_res_iter=unpickle_obj(right_iter_path)
        except FileNotFoundError:
            classification_res_logger.debug("Not found {}, loading from db".format(right_iter_path))
            right_res_iter=[i for i in RightResultsIter(result_path=res_path,classifier=classifier)]
            pickle_obj(right_res_iter,right_iter_path)

        return right_res_iter


class WrongResultsIter(ClassificationResultStream):
    """
    Converts all the text files in a classification result directory that contains
        WRONG/incorrect classification result into ClassificationResultInstance.
        Ignores any files that contains right results, aka the text files ending with _right.txt

    """

    def __init__(self,result_path,classifier=None):
        super().__init__(result_path,classifier)

        self.res_gen=super().get_classification_res_gen("_wrong.txt")

    def __next__(self):
        return next(self.res_gen)

    def __iter__(self):
        return self

    @staticmethod
    def load_iter_from_file(res_path,classifier=None,pickle_path=settings.PICKLE_DIR):
        wrong_iter_path=os.path.join(pickle_path,"wrong_res_iter")

        try:
            wrong_res_iter=unpickle_obj(wrong_iter_path)
        except FileNotFoundError:
            classification_res_logger.debug("Not found {}, loading from db".format(wrong_iter_path))
            wrong_res_iter=[i for i in WrongResultsIter(result_path=res_path,classifier=classifier)]
            pickle_obj(wrong_res_iter,wrong_iter_path)

        return wrong_res_iter
