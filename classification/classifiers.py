import itertools
import re
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

import scipy.sparse as sp
from sklearn.linear_model.logistic import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kN
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC as SV

from util.Logger import Logger
from classification.util import pickle_obj,unpickle_obj

__author__ = 'Kevin'

classifier_logger=Logger()

class Classifier:
    """
    Important formats:
        Coming soon

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
        self.csv_indexes=["ref id","Predicted","Actual"] #output csv column labels
        self.__first_wrong=True #used for keeping track of index
        self.__first_right=True

    def pipeline(self,train_set_iter,test_set_iter,feature_selector,classifiers):
        classifier_logger.info("Starting pipeline for classifier with label {}".format(self.label))

        self.load_training(train_iterable=train_set_iter,maybe_load_vectorizer_from_pickle=True
                           ,maybe_load_training_from_pickle=False,pickle_training=True)\
            .feature_selection(feature_selector)\
            .load_testing(test_iterable=test_set_iter,maybe_load_testing_from_pickle=False,pickle_testing=True).classify(classifiers)


        return self


    def fit_vocab(self,*,train_set_iter=None,load_vectorizer_from_file=False,pickle=True):
        """
        fit vocab to the attr_map of the training set implementing the data set interface. See init.py

        :param train_set_iter: An iterable object with attr_map field
        :return: None
        """

        vocab_loaded=False
        pickle_file=self.pickle_dir+"/classifier_vocab_{}_pickle".format(self.label.split("_")[0])

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

    genre_mapping={}
    #genre_mapping={"Business":"Computers","Shopping":"Computers"}
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
        trainX_pickle_path=self.pickle_dir+"/classifier_trainingX_{}_pickle".format(self.label.split("_")[0])
        trainy_pickle_path=self.pickle_dir+"/classifier_trainingy_{}_pickle".format(self.label.split("_")[0])
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
                global genre_mapping
                #genre mapping allows us to change the distribution of classes
                train_labels.append(self.genre_mapping.get(train_bow_obj.short_genre,train_bow_obj.short_genre))

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


    def load_testing(self,*,test_iterable,maybe_load_testing_from_pickle=False,pickle_testing=False):
        """
        Load the testing set from an iterable of objects with attr_map dictionary attribute and ref_index attribute

        Note: Any feature selection should be done before this step. Or else, we will default to the default vocab
            vectorizer

        :return:
        """

        if self.train_X is None or self.train_y is None:
            raise AttributeError("Train X and Train Y should not be None")

        classifier_logger.info("Loading testing set for classifier {}".format(self.label))

        testing_loaded=False
        testX_pickle_path=self.pickle_dir+"/classifier_testX_{}_pickle".format(self.label.split("_")[0])
        testy_pickle_path=self.pickle_dir+"/classifier_testy_{}_pickle".format(self.label.split("_")[0])
        test_ref_matrix_path=self.pickle_dir+"/classifier_ref_matrix_{}_pickle".format(self.label.split("_")[0])

        test_X=[]
        test_labels=[]
        ref_indexes=[]
        if maybe_load_testing_from_pickle:
            try:
                test_X=unpickle_obj(testX_pickle_path)
                test_labels=unpickle_obj(testy_pickle_path)
                ref_indexes=unpickle_obj(test_ref_matrix_path)
                testing_loaded=True
            except:
                classifier_logger.debug("Failed to load test set from pickle")

        if not testing_loaded:

            explored=set()
            for count,test_obj in enumerate(test_iterable):
                if count %1000==0:
                    classifier_logger.info("Test load curr at:  {}".format(count))

                if test_obj.ref_index in explored:
                    continue

                explored.add(test_obj.ref_index)

                test_labels.append(self.genre_mapping.get(test_obj.short_genre,test_obj.short_genre))
                test_X.append(test_obj.attr_map)
                ref_indexes.append(test_obj.ref_index)

            if pickle_testing:
                classifier_logger.debug("Pickling test files")
                pickle_obj(test_X,testX_pickle_path)
                pickle_obj(test_labels,testy_pickle_path)
                pickle_obj(ref_indexes,test_ref_matrix_path)

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
            try:
                classifier_name=re.search(r"(?<=')(.*)(?=')",str(type(classifier))).group(1).split(".")[-1]

                classifier_logger.info("Classifying with {} with {} training and {} labels".format(classifier_name,
                                                                                                   self.train_X.shape[0],
                                                                                                   self.train_y.shape[0]))
                classifier.fit(self.train_X,self.train_y)

                self.__first_wrong=True
                self.__first_right=True

                for l in range(0,num_labels,increment):
                    res,res_prob_list=classifier.predict_multi(self.test_X[l:l+increment if l+increment<num_labels else num_labels])
                    self.print_res(classifier_name=classifier_name,
                              labels=self.test_y[l:l+increment if l+increment<num_labels else num_labels],
                              predictions=res,
                              ref_indexes=self.test_ref_index_matrix[l:l+increment])

                    #write out the probability
                    with open("ll_prob_1",mode="a") as ll_file:
                        for c,prob in enumerate(res_prob_list):
                            giant_list=itertools.chain(*[(k,v) for k,v in prob.items()])

                            ll_file.write("{}\n".format(','.join([str(self.test_ref_index_matrix[l:l+increment][c])]+[str(i) for i in giant_list])))

            except ValueError:
                pass


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

        wrong_file="{}/{}.txt".format(folder_path,classifier_name+"_wrong")
        right_file="{}/{}.txt".format(folder_path,classifier_name+"_right")
        #1d array, just basic single class prediction, avoids the log likehood predictions
        #as numpy array is not used for those
        if hasattr(predictions,"shape") and len(predictions.shape)==1:
            wrong_labels=(predictions != labels)
        else:
            wrong_labels=np.array([int(l not in preds) for l,preds in zip(labels,predictions)])

        wrong_instances=[]
        right_instances=[]
        for l in range(0,len(wrong_labels)):

            if wrong_labels[l]==1:
                wrong_instances.append([ref_indexes[l],list(predictions[l]),labels[l]])

            else:
                right_instances.append([ref_indexes[l],list(predictions[l]),labels[l]])

        right_data=pd.DataFrame(right_instances,columns=self.csv_indexes)
        wrong_data=pd.DataFrame(wrong_instances,columns=self.csv_indexes)

        right_data.to_csv(right_file,mode="a",index=False,header=self.__first_wrong,columns=self.csv_indexes)
        wrong_data.to_csv(wrong_file,mode="a",index=False,header= self.__first_right,columns=self.csv_indexes)

        self.__first_wrong=False
        self.__first_right=False


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

class BaseClassifier:
    def __init__(self,threshold=1,ll_ranking=False):
        self.threshold=threshold
        self.ll_ranking=ll_ranking

    def predict_multi(self, X):
        """
        Alternative predict function for logistic regression, returns the top predictions for test instances

        :param X:
        :param top_k:
        :return:
        """


        if self.ll_ranking:
            #get the best class based on log likelihood
            predictions=self.predict_proba(X)
            indexes=np.argsort(predictions)
            best_value_indexes=indexes[:,-1:].reshape(-1)
            res_classes=[]
            best_values=predictions[list(range(0,best_value_indexes.shape[0])),best_value_indexes]

            for c,row in enumerate(predictions):
                res_classes.append(self.classes_[[c for c,ll in enumerate(row) if best_values[c]-ll<=self.threshold]])

        else:
            assert self.threshold<self.classes_.shape[0]

            predictions=self.predict_proba(X)
            top_x_indexes=np.argsort(predictions)[:,:-(self.threshold+1):-1]

            res_classes=self.classes_[top_x_indexes]

        class_to_prob_list=[]
        #construction a list of dictionary of classes to their likelihood values
        for p_row in predictions:
            class_to_prob_list.append({c:p for (p,c) in zip(p_row,self.classes_)})

        return res_classes,class_to_prob_list


class LogisticRegression(LR,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        LR.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)


class kNN(kN,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        kN.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)


class mNB(MB,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        MB.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)

class RandomForest(RF,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        RF.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)

class DecisionTree(DT,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        DT.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)

class SVC(SV,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        SV.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)