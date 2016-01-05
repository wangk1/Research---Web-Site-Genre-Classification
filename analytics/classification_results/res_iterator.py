import os
import re
import ast

import pandas as pd

import settings
import util.Logger as Logger
from db.db_model.mongo_websites_models import URLBow
from util.base_util import normalize_genre_string
from data.util import unpickle_obj,pickle_obj

classification_res_logger=Logger.Logger()


"""
This module is geared towards providing functions that transforms the classification result file.

Current functions:
    ClassificationInstance, namedtuple that represent an entry in the classficiation result file
    get_classification_res, function that transforms classfication result file to actual ClassificationInstance objs.

"""

__author__ = 'Kevin'


CLASSIFICATION_HEADERS=["ref id","Predicted","Actual"]

class ClassificationResultInstance:

    def __init__(self,ref_id,actual,predicted,classifier,genre_lv=1):
        self.ref_id=ref_id
        self.__actual=[actual] if isinstance(actual,str) else actual
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

        #isolate the name of the classifier
        classifier=re.search(".*(?=_cres.txt)",filepath.split("\\")[-1]).group(0)

        #file_datagram=pd.read_csv(filepath)

        #for f in file_datagram.values:

        with open(filepath) as res_file:
            next(res_file) #burn first line, the column headings
            for line in res_file:
                line=line[:-1]

                lines=re.split(",(?= *\[)",line)

                prediction_objs.append(ClassificationResultInstance(ref_id=lines[0],predicted=ast.literal_eval(lines[1])
                                                                ,actual=ast.literal_eval(lines[2]),
                                                        classifier=classifier))

        return prediction_objs

    def get_classification_res_gen(self,suffix="cres.txt"):
        """
        Generator that generates the next classification result entry.

        It will read files with only specified suffix, so this can be used to filter
            out the right files or the wrong files or vice versa or any files really.

        :return:
        """
        abs_result_path=os.path.abspath(self.result_path)

        #in case we only chose some classifiers
        if self.classifiers is not None:
            res_files=(f for f in os.listdir(abs_result_path)
                         if any(f=="_".join((c,suffix)) for c in self.classifiers))

        else:
            res_files=(f for f in os.listdir(abs_result_path) if f=="_".join((self.classifiers[0],suffix)))

        for res_file in res_files:
            for res_obj in self.get_classification_res(os.path.join(abs_result_path,res_file)):
                yield res_obj

        raise StopIteration("End of right result stream")

def default_right_comp(pred,actual):
    """
    Function that determines if the predicted class matches actual.

    This is the default used. It compares the top prediction at index 0 in the pred parameter to
        each element in actual. If it matches one, it is a hit.

    :param pred:
    :param actual:
    :return:
    """

    return pred in actual

def default_pred_transformer(pred):
    """
    Default transformer for predictions, we only want the top prediction to compare with actual.

    :param pred:
    :return:
    """
    return pred[0]

class RightResultsIter(ClassificationResultStream):
    """
    Converts all the text files in a classification result directory that contains
        RIGHT/correct classification result into ClassificationResultInstance.
        Ignores any files that contains wrong results, aka the text files ending with _wrong.txt

    """

    def __init__(self,result_path,classifier=None,secondary_identifier="",comparator=default_right_comp
                 ,pred_tranformer=default_pred_transformer,actual_transformer=lambda x:x):
        super().__init__(result_path,classifier)

        self.res_gen=super().get_classification_res_gen("{}_cres.txt".format(secondary_identifier) if secondary_identifier else "cres.txt")
        self.is_right=comparator
        self.pred_transformer=pred_tranformer
        self.actual_transformer=actual_transformer

    def __next__(self):
        pred_obj= next(self.res_gen)

        #check to see if actually a right instance
        while not self.is_right(self.pred_transformer(pred_obj.predicted),self.actual_transformer(pred_obj.actual)):
            pred_obj=next(self.res_gen)

        return pred_obj

    def __iter__(self):
        return self


class WrongResultsIter(ClassificationResultStream):
    """
    Converts all the text files in a classification result directory that contains
        WRONG/incorrect classification result into ClassificationResultInstance.
        Ignores any files that contains right results, aka the text files ending with _right.txt

    """

    def __init__(self,result_path,classifier=None,secondary_identifier="",comparator=lambda p,a: not default_right_comp(p,a)
        ,pred_tranformer=default_pred_transformer,actual_transformer=lambda x:x):

        super().__init__(result_path,classifier)

        self.res_gen=super().get_classification_res_gen("{}_cres.txt".format(secondary_identifier) if secondary_identifier else "cres.txt")
        self.is_wrong=comparator
        self.pred_transformer=pred_tranformer
        self.actual_transformer=actual_transformer

    def __next__(self):
        pred_obj= next(self.res_gen)

        #check to see if actually a right instance
        while not self.is_wrong(self.pred_transformer(pred_obj.predicted),self.actual_transformer(pred_obj.actual)):
            pred_obj=next(self.res_gen)

        return pred_obj

    def __iter__(self):
        return self
