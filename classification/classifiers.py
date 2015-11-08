import itertools
import re

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp

from util.Logger import Logger
from classification.util import pickle_obj,unpickle_obj

__author__ = 'Kevin'

classifier_logger=Logger()

class Classifier:
    """
    Class used as a pipeline for classification. Makes things easier to manager

    An instance of this class is used to keep track of the state of classification

    1. Be able to save the fitted attribute vectorizer to disk to conserve memory and avoid reloading vocab each time
    2. Likewise for training and testing samples
    3. Keep track of the outputting location of the classification result files

    """
    def __init__(self,label,*,res_dir,attr_vectorizer_dir):
        """
        :param label: Unique label
        :param res_dir:
        :param attr_vectorizer_dir:
        :return:
        """
        self.label=label

        self.res_dir=res_dir
        self.attr_vectorizer_dir=attr_vectorizer_dir

        self.vocab_vectorizer=None

        self.train_X=None
        self.train_y=None

        self.test_X=None
        self.test_y=None
        self.test_ref_index_matrix=None #unique id of nx1 where n is the number of samples in test set.

    def pipeline(self,train_set_iter,test_set_iter,feature_selector,classifiers):
        classifier_logger.info("Starting pipeline for classifier with label {}".format(self.label))

        self.load_training(train_iterable=train_set_iter).load_testing(test_iterable=test_set_iter) \
            .feature_selection(feature_selector).classify(classifiers)

        return self


    def fit_vocab(self,*,train_set_iter=None,load_vectorizer_from_file=False,pickle=True):
        """
        fit vocab to the attr_map of the training set implementing the data set interface. See init.py

        :param train_set_iter: An iterable object with attr_map field
        :return: None
        """

        vocab_loaded=False
        pickle_file=self.attr_vectorizer_dir+"/classifier_vocab_{}_pickle".format(self.label)

        if load_vectorizer_from_file:
            try:
                unpickle_obj(pickle_file)

                vocab_loaded=True
            except FileNotFoundError:
                classifier_logger.info("Failed to find pickle file {}".format(pickle_file))
                vocab_loaded=False

        if not vocab_loaded:

            classifier_logger.info("Fitting vocab for classifier {}".format(self.label))

            train_dv=DictVectorizer()

            words=[dict(itertools.chain(*(train_set_obj.attr_map.items() for train_set_obj in train_set_iter)))]
            #fit the dv first
            train_dv.fit(words)
            classifier_logger.info("vocab length is {}".format(len(train_dv.feature_names_)))

            del words

            #pickle the obj if we want to
            pickle and pickle_obj(train_dv,pickle_file)


        return self


    def load_training(self,train_iterable,stack_per_sample=3000):
        """
        Load the training set with attr_map dictionary attribute and return a scipy sparse matrix of the data fitted
            with the vocab and their labels

        :returns: train_X: the data matrix. train_y: the train set labels

        """

        if self.train_X is not None:
            classifier_logger.info("Reloading training samples")

        if self.vocab_vectorizer is None:
            self.fit_vocab(train_set_iter=(obj for obj in train_iterable))

        classifier_logger.info("Loading training set for classifier pipeline {}".format(self.label))


        train_labels=[]
        matrix_cache=[]
        for count,train_bow_obj in enumerate(train_iterable):
            if count %1000==0:
                classifier_logger.info("Train load curr at:  {}".format(count))

            curr_bow_matrix=self.vocab_vectorizer.transform(train_bow_obj.attr_map)[0]
            matrix_cache.append(curr_bow_matrix)
            train_labels.append(train_bow_obj.short_genre)

            if len(matrix_cache)>stack_per_sample:
                self.train_X=sp.vstack(matrix_cache)
                matrix_cache=[self.train_X]
                classifier_logger.info("stacked, train bow size:{},labels size: {}".format(
                    self.train_X.shape[0],len(train_labels)))

        if len(matrix_cache)>0:
            classifier_logger.info("stacking")
            self.train_X=sp.vstack(matrix_cache)
            del matrix_cache

        self.train_y=np.asarray(train_labels)

        classifier_logger.info("Final training size: {}".format(self.train_X.shape[0]))
        return self


    def load_testing(self,*,test_iterable):
        """
        Load the testing set from an iterable of objects with attr_map dictionary attribute and ref_index attribute

        :return:
        """

        if self.train_X is None or self.train_y is None:
            raise AttributeError("Train X and Train Y should not be None")

        classifier_logger.info("Loading testing set for classifier {}".format(self.label))

        test_X=[]
        test_labels=[]
        ref_indexes=[]

        explored=set()
        for count,test_obj in enumerate(test_iterable):
            if count %1000==0:
                classifier_logger.info("Test load curr at:  {}".format(count))

            if test_obj.ref_index in explored:
                continue

            explored.add(test_obj.ref_index)

            test_labels.append(test_obj.short_genre)
            test_X.append(test_obj.attr_map)
            ref_indexes.append(test_obj.ref_index)


        self.test_X=self.vocab_vectorizer.transform(test_X)
        self.test_y=np.asarray(test_labels)
        self.test_ref_index_matrix=np.asarray(ref_indexes)

        return self

    def classify(self,classifiers,increment=500):
        """
        Classify with test_X and generate result files for each classifier set

        :param classifiers: set of additional classifier to test with in addition to self.classifier classifiers
        :return:
        """
        num_labels=len(self.test_y)

        for classifier in classifiers:
            classifier_logger.info("Classifying with {}".format(str(type(classifier))))

            for l in range(0,num_labels,increment):
                res=classifier.predict(self.test_X[l:l+increment if l+increment<num_labels else num_labels])
                self.print_res(classifier_name=re.search(r"(?<=')(.*)(?=')",str(type([]))).group(1),
                          labels=num_labels[l:l+increment if l+increment<num_labels else num_labels],
                          predictions=res,
                          ref_indexes=self.test_ref_index_matrix[l:l+increment])

    def print_res(self,classifier_name,predictions,labels,ref_indexes):
        """
        Compare the prediction to the actual labels and print them into the appropriate *_wrong.txt or *_right.txt in
               their respective result directories

        :param classifier_name:
        :param predictions:
        :param labels:
        :param ref_indexes:
        :return:
        """
        with open("{}/{}/{}.txt".format(self.res_dir,self.label,classifier_name+"_wrong"),mode="a") as out_wrong, \
            open("{}/{}/{}.txt".format(self.res_dir,self.label,classifier_name+"_wrong").format(classifier_name+"_right"),mode="a") as out_right:
            wrong_labels=(predictions != labels)

            for l in range(0,len(wrong_labels)):
                if wrong_labels[l]==1:
                    out_wrong.write("{}, predicted: {}, actual: {}\n".format(ref_indexes[l],predictions[l],labels[l]))
                else:
                    out_right.write("{}, predicted: {}, actual: {}\n".format(ref_indexes[l],predictions[l],labels[l]))


    def feature_selection(self,feat_selector):
        """
        Perform feature selection

        :param feat_selector:
        :return:
        """

        assert hasattr(feat_selector,"transform")

        self.train_X=feat_selector.fit_transform(self.train_X,self.train_y)

        return self