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
from data.util import pickle_obj,unpickle_obj

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
    def __init__(self):
        self.csv_indexes=["ref id","Predicted","Actual"] #output csv column labels
        self.res_dir="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\supervised"


    def classify(self,learn_settings,train_set,test_set,classifiers,increment=500):
        """
        Classify with test_X and generate result files for each classifier set

        :param classifiers: set of additional classifier to test with in addition to self.classifier classifiers
        :return:
        """
        for classifier in classifiers:

            classifier_name=re.search(r"(?<=')(.*)(?=')",str(type(classifier))).group(1).split(".")[-1]

            classifier_logger.info("Classifying with {} with {} training and {} labels".format(classifier_name,
                                                                                               train_set.X.shape[0],
                                                                                               train_set.y.shape[0]))
            classifier.fit(train_set.X,train_set.y)

            res,res_prob_list=classifier\
                    .predict_multi(test_set.X)

            print("Done, printing Results for {}".format(classifier_name))
            self.print_res(learn_settings,
                  y=test_set.y,
                  predictions=res,
                  ref_indexes=test_set.ref_indexes,
                  classifier_name=classifier_name)

                # #write out the probability
                # with open("ll_prob_1",mode="a") as ll_file:
                #     for c,prob in enumerate(res_prob_list):
                #         giant_list=itertools.chain(*[(k,v) for k,v in prob.items()])
                #
                #         ll_file.write("{}\n".format(','.join([str(self.test_ref_index_matrix[l:l+increment][c])]+[str(i) for i in giant_list])))





    def print_res(self,learn_settings,y,ref_indexes,predictions,classifier_name):
        """
        Compare the prediction to the actual labels and print them into the appropriate *_wrong.txt or *_right.txt in
               their respective result directories

        :param classifier_name:
        :param predictions:
        :param labels:
        :param ref_indexes:
        :return:
        """
        folder_path="{}/{}".format(self.res_dir,learn_settings.label)

        #make dir if not exist
        os.makedirs(folder_path,exist_ok=True)

        output_file="{}/{}_cres.txt".format(folder_path,classifier_name)

        #emit the csv
        with open(output_file,mode="w") as output_handler:
            output_handler.write(",".join(self.csv_indexes)+"\n")
            for l in range(0,y.shape[0]):
                output_handler.write("{},{},{}\n".format(ref_indexes[l],list(predictions[l]),list(y[l])))



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