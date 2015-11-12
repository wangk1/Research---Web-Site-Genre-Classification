import collections
import os
import re
"""
This module is geared towards providing functions that transforms the classification result file.

Current functions:
    ClassificationInstance, namedtuple that represent an entry in the classficiation result file
    get_classification_res, function that transforms classfication result file to actual ClassificationInstance objs.

"""

__author__ = 'Kevin'

ClassificationResultInstance=collections.namedtuple("ClassificationInstance",("ref_id","actual","predicted","classifier"))

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

class ClassificationResultStream:
    """
    Base class for streaming prediction results from the text files, regardless of the results
    being wrongly predicted for correctly predicted

    Most of the time, the subclasses RightResultsStreamIter or WrongResultsStreamIter should be used
    """

    def __init__(self,result_path,classifier=None):
        """
        Read results from the path of the results and prepare the iterator.

        The files in the result folder should follow the classification result file interface

        :param result_path: The path to folder containing all the classifier results
        :param classifier: Allow use to only stream the data from the results of one classifier.
            Current this feature is not supported.
        """
        assert os.path.isdir(result_path) and os.path.exists(result_path)

        self.result_path=result_path

        if classifier is not None:
            raise NotImplementedError("Current streaming by classifier does not exist")

    def get_classification_res(self,filepath):
        """
        Convert a result file into the ClassificationResultInstance objects

        :return: list of ClassificationInstance objects that represents an id of webpage and its actual, predicted, and classifier.
        """
        prediction_objs=[]

        classifier=re.search(".*(?=_(wrong|right).txt)",filepath.split("\\")[-1]).group(0)
        with open(filepath) as sample_file:
            for line in sample_file:

                res_list=line.split(" ")

                prediction_objs.append(ClassificationResultInstance(ref_id=res_list[0][:-1],predicted=res_list[2][:-1],actual=res_list[4][:-1],
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
