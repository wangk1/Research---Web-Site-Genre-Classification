import itertools
import re
import os
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
    def __init__(self,label,*,res_dir,vocab_vectorizer_pickle_dir):
        """
        :param label: Unique label
        :param res_dir: The main directory to put all classification results in
        :param vocab_vectorizer_pickle_dir: The main directory to leave all pickled vocabulary vectorizer
        :return:
        """
        self.label=label

        self.res_dir=res_dir
        self.pickle_dir=vocab_vectorizer_pickle_dir

        self.vocab_vectorizer=None
        self.feat_selector=None

        self.train_X=None
        self.train_y=None

        self.test_X=None
        self.test_y=None
        self.test_ref_index_matrix=None #unique id of nx1 where n is the number of samples in test set.

    def pipeline(self,train_set_iter,test_set_iter,feature_selector,classifiers):
        classifier_logger.info("Starting pipeline for classifier with label {}".format(self.label))

        self.load_training(train_iterable=train_set_iter,maybe_load_vectorizer_from_pickle=True
                           ,maybe_load_training_from_pickle=True,pickle_training=True)\
            .feature_selection(feature_selector)\
            .load_testing(test_iterable=test_set_iter).classify(classifiers)


        return self


    def fit_vocab(self,*,train_set_iter=None,load_vectorizer_from_file=False,pickle=True):
        """
        fit vocab to the attr_map of the training set implementing the data set interface. See init.py

        :param train_set_iter: An iterable object with attr_map field
        :return: None
        """

        vocab_loaded=False
        pickle_file=self.pickle_dir+"/classifier_vocab_{}_pickle".format(self.label)

        if load_vectorizer_from_file:
            try:
                self.vocab_vectorizer=unpickle_obj(pickle_file)

                vocab_loaded=True
                classifier_logger.info("Loaded pickle file {}".format(pickle_file))
            except FileNotFoundError:
                classifier_logger.info("Failed to find pickle file {}".format(pickle_file))
                vocab_loaded=False

        if not vocab_loaded:

            classifier_logger.info("Fitting vocab for classifier {}".format(self.label))

            self.vocab_vectorizer=DictVectorizer()

            words=[dict(itertools.chain(*(train_set_obj.attr_map.items() for train_set_obj in train_set_iter)))]
            #fit the dv first
            self.vocab_vectorizer.fit(words)
            classifier_logger.info("vocab length is {}".format(len(self.vocab_vectorizer.feature_names_)))

            del words

            #pickle the obj if we want to
            pickle and pickle_obj(self.vocab_vectorizer,pickle_file)


        return self


    def load_training(self,train_iterable,stack_per_sample=3000,maybe_load_vectorizer_from_pickle=False,
                      maybe_load_training_from_pickle=False,pickle_training=False):
        """
        Load the training set with attr_map dictionary attribute and return a scipy sparse matrix of the data fitted
            with the vocab and their labels

        :returns: train_X: the data matrix. train_y: the train set labels

        """

        if self.train_X is not None:
            classifier_logger.info("Reloading training samples")

        if self.vocab_vectorizer is None:
            self.fit_vocab(train_set_iter=(obj for obj in train_iterable),load_vectorizer_from_file=maybe_load_vectorizer_from_pickle)

        classifier_logger.info("Loading training set for classifier pipeline {}".format(self.label))

        training_loaded=False
        trainX_pickle_path=self.pickle_dir+"/classifier_trainingX_{}_pickle".format(self.label)
        trainy_pickle_path=self.pickle_dir+"/classifier_trainingy_{}_pickle".format(self.label)
        if maybe_load_training_from_pickle:
            classifier_logger.info("Trying to load training samples from pickle")
            try:
                self.train_X=unpickle_obj(trainX_pickle_path)
                self.train_y=unpickle_obj(trainy_pickle_path)
                training_loaded=True
                classifier_logger.info("Successfully loaded training samples from pickle")
            except FileNotFoundError:
                classifier_logger.info("Failed to load training samples from pickle")

        if not training_loaded:
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

            #pickle if so
            if pickle_training:
                classifier_logger.info("Pickling training set")
                pickle_obj(self.train_X,trainX_pickle_path)
                pickle_obj(self.train_y,trainy_pickle_path)
                classifier_logger.info("Successfully save training samples from pickle")

        classifier_logger.info("Final training size: {}".format(self.train_X.shape[0]))
        return self


    def load_testing(self,*,test_iterable):
        """
        Load the testing set from an iterable of objects with attr_map dictionary attribute and ref_index attribute

        Note: Any feature selection should be done before this step. Or else, we will default to the default vocab
            vectorizer

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


        self.test_X=self.vocab_vectorizer.transform(test_X) if self.feat_selector is None else \
            self.feat_selector.transform(self.vocab_vectorizer.transform(test_X))
        self.test_y=np.asarray(test_labels)
        self.test_ref_index_matrix=np.asarray(ref_indexes)

        classifier_logger.info("Testing load successful, num of samples: {}, number of features {}".format(*self.test_X.shape))

        return self

    def classify(self,classifiers,increment=500):
        """
        Classify with test_X and generate result files for each classifier set

        :param classifiers: set of additional classifier to test with in addition to self.classifier classifiers
        :return:
        """
        num_labels=len(self.test_y)

        for classifier in classifiers:
            classifier_name=re.search(r"(?<=')(.*)(?=')",str(type(classifier))).group(1).split(".")[-1]

            classifier_logger.info("Classifying with {} with {} training and {} labels".format(classifier_name,
                                                                                               self.train_X.shape[0],
                                                                                               self.train_y.shape[0]))
            classifier.fit(self.train_X,self.train_y)

            for l in range(0,num_labels,increment):
                res=classifier.predict(self.test_X[l:l+increment if l+increment<num_labels else num_labels])
                self.print_res(classifier_name=classifier_name,
                          labels=self.test_y[l:l+increment if l+increment<num_labels else num_labels],
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
        folder_path="{}/{}".format(self.res_dir,self.label)
        if not os.path.exists(folder_path):
            classifier_logger.debug("Creating folder {}".format(folder_path))
            os.mkdir(folder_path)

        with open("{}/{}.txt".format(folder_path,classifier_name+"_wrong"),mode="a") as out_wrong, \
            open("{}/{}.txt".format(folder_path,classifier_name+"_right"),mode="a") as out_right:
            wrong_labels=(predictions != labels)

            for l in range(0,len(wrong_labels)):
                if wrong_labels[l]==1:
                    out_wrong.write("{}, predicted: {}, actual: {}\n".format(ref_indexes[l],predictions[l],labels[l]))
                else:
                    out_right.write("{}, predicted: {}, actual: {}\n".format(ref_indexes[l],predictions[l],labels[l]))


    def feature_selection(self,feat_selector):
        """
        Perform feature selection. Must be done before loading testing sets

        :param feat_selector:
        :return:
        """

        assert hasattr(feat_selector,"transform")

        classifier_logger.info("Pre feature selection: num features: {}".format(self.train_X.shape[1]))
        self.feat_selector=feat_selector
        self.train_X=feat_selector.fit_transform(self.train_X,self.train_y)
        classifier_logger.info("Post feature selection: num features: {}".format(self.train_X.shape[1]))

        return self