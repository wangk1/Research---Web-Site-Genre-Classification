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
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AC
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
        self.res_dir="classification_res\\supervised"


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

            classifier_logger.info("Fitting done, predicting with test set")

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

        output_file="{}/{}{}_cres.txt".format(folder_path,classifier_name,"" if learn_settings.result_file_label=="" else "_"+learn_settings.result_file_label)

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

    def predict_multi(self, X,predictions_per_stack=1500):
        """
        Alternative predict function for logistic regression, returns the top predictions for test instances

        :param X:
        :param top_k:
        :return:
        """


        if self.ll_ranking:
            #get the best class based on log likelihood
            predictions_classes=self.predict_proba(X)
            indexes=np.argsort(predictions_classes)
            best_value_indexes=indexes[:,-1:].reshape(-1)
            res_classes=[]
            best_values=predictions_classes[list(range(0,best_value_indexes.shape[0])),best_value_indexes]

            for c,row in enumerate(predictions_classes):
                res_classes.append(self.classes_[[c for c,ll in enumerate(row) if best_values[c]-ll<=self.threshold]])

        else:
            assert self.threshold<self.classes_.shape[0]

            predictions_classes=None
            for c,r in enumerate(range(0,X.shape[0],predictions_per_stack)):
                print("On Stack {}".format(c))

                X_slice=X[r:r+predictions_per_stack if r+predictions_per_stack<X.shape[0] else X.shape[0]]
                prediction=self.predict_proba(X_slice)
                top_x_indexes=np.argsort(prediction)[:,:-(self.threshold+1):-1]

                res_classes=self.classes_[top_x_indexes]

                predictions_classes=np.vstack((predictions_classes,res_classes)) if predictions_classes is not None else res_classes

        print("Total number of predictions: {}".format(predictions_classes.shape[0]))

        class_to_prob_list=[]
        #construction a list of dictionary of classes to their likelihood values
        #for p_row in predictions:
            #class_to_prob_list.append({c:p for (p,c) in zip(p_row,self.classes_)})

        return predictions_classes,class_to_prob_list

class MultiClassifier:
    """
    Represents combination of multiple classifiers

    """
    def __init__(self,classifiers,weights=[]):
        self.classifiers=classifiers
        self.weights=weights

    def fit(self,train_X,train_y):
        """
        Fit the training set to each classifier.

        :param train_X: A list or iterable of training X for each classifier
        :param train_y: A numpy array or similar of labels. Note that there is only 1 label set for all train_X's
        :return:
        """
        assert len(train_X)==len(self.classifiers)

        for num,classifier in enumerate(self.classifiers):
            classifier_logger.info("Fitting with {} on dataset {}".format(classifier,num))

            classifier.fit(train_X[num],train_y)


    def predict_multi(self,test_X):
        """
        Predict the individual test label with their respective classifier

        :param test_X:
        :return:
        """
        assert len(test_X) == len(self.classifiers)

        predictions=[]
        for classifier in self.classifiers:
            predictions.append(classifier.predict_multi(test_X))


        pass

class LogisticRegression(LR,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        LR.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)


class kNN(kN,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        kN.__init__(self,**kwargs)
        BaseClassifier.__init__(self,threshold=threshold,ll_ranking=ll_ranking)

class Ada(AC,BaseClassifier):
    def __init__(self,threshold=1,ll_ranking=False,**kwargs):
        AC.__init__(self,**kwargs)
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