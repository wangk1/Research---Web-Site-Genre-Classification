import itertools
import re
import os
import statistics

import numpy as np
import pandas as pd
import collections as coll
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




                # #write out the probability
                # with open("ll_prob_1",mode="a") as ll_file:
                #     for c,prob in enumerate(res_prob_list):
                #         giant_list=itertools.chain(*[(k,v) for k,v in prob.items()])
                #
                #         ll_file.write("{}\n".format(','.join([str(self.test_ref_index_matrix[l:l+increment][c])]+[str(i) for i in giant_list])))



    def print_cross_validation_res(self,cv_results):

        for classifiers_name in cv_results.results:
            average=statistics.mean((res_single.accuracy for fold,res_single in cv_results.results[classifiers_name].items()))

            print("")



    def print_res(self,learn_settings,y,ref_indexes,predictions,classifier_name,shorten_path=False):
        """
        Compare the prediction to the actual labels and print them into the appropriate *_wrong.txt or *_right.txt in
               their respective result directories

        Shorten path is due to the fact that python glitches out if the number of characters in the filename is >65 char
            So, we can fix the issue by just shortening each component of the filename to 3 characters
        :param classifier_name:
        :param predictions:
        :param labels:
        :param ref_indexes:
        :return:
        """
        if shorten_path:
            classifier_name=classifier_name[:4]

        label="_".join([setting.label for setting in learn_settings] if isinstance(learn_settings,coll.Iterable) else learn_settings.label)

        folder_path="{}/{}".format(self.res_dir,label)

        #make dir if not exist
        os.makedirs(folder_path,exist_ok=True)

        pwd=os.getcwd() #change to the other directory so the file path isn't excessively long
        os.chdir(folder_path)

        output_file="{}{}_cres.txt".format(classifier_name,"" if learn_settings[0].result_file_label=="" else "_"+learn_settings[0].result_file_label)

        #emit the csv
        with open(output_file,mode="w") as output_handler:
            output_handler.write(",".join(self.csv_indexes)+"\n")
            for l in range(0,y.shape[0]):
                output_handler.write("{},{},{}\n".format(ref_indexes[l],list(predictions[l]),list(y[l])))

        os.chdir(pwd)

    def calculate_accuracy(self,label,predictions,comparator=lambda l,p: p in l):
        """
        Calculate accuracy of prediction

        :param label:
        :param predictions:
        :param comparator:
        :return:
        """

        return round(sum((comparator(l,predictions[c]) for c,l in enumerate(label)))/label.shape[0],3)

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return re.search(r"(?<=')(.*)(?=')",str(type(self))).group(1).split(".")[-1]

    def predict_proba(self,X):
        raise NotImplementedError("Not implemented")

    def predict_multi(self, X,predictions_per_stack=1000,return_prediction_prob=False):
        """
        Alternative predict function for logistic regression, returns the top predictions for test instances

        :param X:
        :param top_k:
        :return:
        """
        prediction_res=None
        predictions_probs=None
        predictions_classes=None

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


            for c,r in enumerate(range(0,X.shape[0],predictions_per_stack)):
                print("On Stack {}".format(c))

                X_slice=X[r:r+predictions_per_stack if r+predictions_per_stack<X.shape[0] else X.shape[0]]
                prediction_prob=self.predict_proba(X_slice)

                if return_prediction_prob:
                    predictions_probs=np.vstack((predictions_probs,prediction_prob)) \
                        if predictions_probs is not None else prediction_prob


                else:
                    top_x_indexes=np.argsort(prediction_prob)[:,:-(self.threshold+1):-1]

                    res_classes=self.classes_[top_x_indexes]
                    predictions_classes=np.vstack((predictions_classes,res_classes)) if predictions_classes is not None else res_classes

            prediction_res=predictions_classes if not return_prediction_prob else predictions_probs

        print("Total number of predictions: {}".format(prediction_res.shape[0]))

        return prediction_res

class MultiClassifier:
    """
    Represents combination of multiple classifiers

    """
    def __init__(self,classifiers,weights=None,**kwargs):
        self.classifiers=classifiers
        self.weights=weights
        self.classes_=None
        self.threshold=kwargs["threshold"]
        self.ll_ranking=kwargs["ll_ranking"]

        #cache the previous result, makes setting different weights fast
        self.prev_cache=None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "-".join((str(c) for c in self.classifiers))

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

        self.classes_=self.classifiers[0].classes_


    def predict_proba(self,test_Xs,classifier_weights,use_prev_prob):
        """
        Predict the individual test label with their respective classifier

        :param test_Xs:
        :return:
        """
        assert len(test_Xs) == len(self.classifiers)

        self.classes_=self.classifiers[0].classes_
        if use_prev_prob:
            print("Using Previous Predicted Probabiliyt")

            prediction_probs=classifier_weights[0]*self.prev_cache[0]

            for c,(classifier,test_X) in enumerate(zip(self.classifiers[1:],test_Xs[1:]),1):
                assert (classifier.classes_==self.classes_).all()

                prediction_probs+=classifier_weights[c]*self.prev_cache[c]

        else:
            print("Not Using Previous Predicted Probability")

            self.prev_cache=[]

            self.prev_cache.append(self.classifiers[0].predict_multi(test_Xs[0],return_prediction_prob=True))
            prediction_probs=classifier_weights[0]*self.prev_cache[0]


            for c,(classifier,test_X) in enumerate(zip(self.classifiers[1:],test_Xs[1:]),1):
                assert (classifier.classes_==self.classes_).all()

                self.prev_cache.append(classifier.predict_multi(test_X,return_prediction_prob=True))
                prediction_probs+=classifier_weights[c]*self.prev_cache[c]


        return prediction_probs

    def predict_multi(self,X,classifier_weights,use_prev_prob=False):

        """
        Alternative predict function for logistic regression, returns the top predictions for test instances

        :param X:
        :param top_k:
        :return:
        """
        predictions_probs=self.predict_proba(X,classifier_weights,use_prev_prob)

        if self.ll_ranking:
            #get the best class based on log likelihood
            raise NotImplementedError("LL Ranking not implemented")

        else:
            top_x_indexes=np.argsort(predictions_probs)[:,:-(self.threshold+1):-1]

            predictions_classes=self.classes_[top_x_indexes]


        return predictions_classes

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