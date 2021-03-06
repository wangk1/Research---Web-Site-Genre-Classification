import functools,collections as coll
import itertools
import re
import random
from classification.classification_settings import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from classification.classifiers import ClassifierUtil
import util.base_util as util,numpy as np

from sklearn.cross_validation import KFold,ShuffleSplit
from collections import namedtuple
from db.db_model.mongo_websites_classification import URLAllGram,TestSet_urlAllGram,TrainSet_urlAllGram,URLBow_fulltxt, \
    TrainSet_urlFullTextBow,TestSet_urlFullTextBow
import classification.classifiers as classifiers
from data.training_testing import MultiData
from data.util import unpickle_obj
from classification.classification import feature_selection
from functools import partial
from util.base_util import normalize_genre_string
from util.genre import filter_genres
from util.Logger import Logger
from data.X_y import match_sets_based_on_ref_id
from classification.classification import classify, load_training_testing
import operator as op
from classification.results import ResCrossValidation

__author__ = 'Kevin'


supervised_logger=Logger()

genre_dict={'Sports': 8757,
            'Business': 8553,
            'Shopping': 6920,
            'Computers': 6245,
            'Arts': 6165,
            'Society': 5841,
            'Recreation': 5770,
            'Health': 5418,
            'Science': 3662,
            'Games': 2767,
            'Reference': 2219,
            'Kids': 2142,
            'News': 1954,
            'Regional': 1949,
            'Home': 1929,
            'Adult': 1846,
            }



ClassificationSource=namedtuple("Source",("ref_index","attr_map","short_genre"))
def generate_training_testing(source,test_set_indexes,genres,*,train_coll_cls,test_coll_cls):
    """
    Takes a list of test set indexes and genres.

    The source is an object that has the following interface:
        ref_index: field that is usually a reference to a element in one of the queues
        attr_map: mapping from attribute to count or tfidf
        short_genres: genres for the object


    :param test_set_indexes:
    :param genres: a list or set or tuple of the genres we are selecting testing set from
    :param source: the source, a mapping from db to ClassificationSource namedtuple for standarization purposes
    :return:
    """

    #track the indexes in each
    index_tracker=dict((k,0) for k in genres)

    BOW_CUTOFF=30
    for c,source_obj in enumerate(source):
        assert isinstance(source_obj,ClassificationSource)

        if c%1000==0:
            print("On url number {}, index: {}".format(c,source_obj.ref_index))

        if source_obj.short_genre not in index_tracker:
            continue

        curr_index=index_tracker[source_obj.short_genre]

        #this is a training set or testing set
        if curr_index in test_set_indexes[source_obj.short_genre]:
            test_coll_cls(**source_obj._asdict()).save()

        else:
            train_coll_cls(**source_obj._asdict()).save()

        index_tracker[source_obj.short_genre]+=1



def get_full_text_genres():
    genres_iter=(set(util.normalize_genre_string(g,1) for g in url_bow_obj.short_genres) for url_bow_obj in URLBow_fulltxt.objects)

    return itertools.chain(*genres_iter)


def map_urlFullText(genre_dict):


    generate_training_testing((ClassificationSource(url_fulltxt_obj.bow_index,url_fulltxt_obj.bow,
                                                    util.normalize_genre_string(url_fulltxt_obj.short_genres[0],1))
                                    for url_fulltxt_obj in URLBow_fulltxt.objects)
                              ,test_set_nums,genre_dict.keys()
                              ,train_coll_cls=TrainSet_urlFullTextBow,test_coll_cls=TestSet_urlFullTextBow)



def cross_validate_gen(total_num_ele,k_folds):
    """
    Generator for generating folds for cross validation

    :param total_num_ele:
    :param k_folds:
    :return:
    """
    kf=KFold(total_num_ele,n_folds=k_folds,shuffle=True,random_state=24829555)

    for train_ind, test_ind in kf:
        yield train_ind,test_ind

def randomized_cross_validation_gen(total_num_ele,k_folds):
    """
    Cross validation stradegy that first randomly shuffles the indices. Then,
    it divides the set into X folds

    :param total_num_ele:
    :param k_folds:
    :return:
    """
    shuffled_indexes=random.sample(range(0,total_num_ele),total_num_ele)

    num_per_fold=total_num_ele//k_folds
    for fold in range(0,k_folds):
        test_end_fold=(fold+1)*num_per_fold if fold != k_folds-1 else len(shuffled_indexes)
        test_start_fold=fold*num_per_fold

        test_ind=shuffled_indexes[test_start_fold:test_end_fold]
        train_ind=shuffled_indexes[:test_start_fold]+shuffled_indexes[test_end_fold:]

        yield train_ind,test_ind



if __name__=="__main__":
    #See classification.classification_settings for the adjustable settings


    supervised_logger.info("Number of Weights: {}".format(weights.num_classifiers))

    #CLASSIFICATION, adjust weights
    classifier_util=ClassifierUtil()

    """
    LOAD DATA, preprocess
    """

    #WARNING: REF INDEX for each individual X set must match row to row
    Xs=[]
    ys=[]
    ref_indexes_unmatched=[]
    ref_indexes=[]

    for setting in settings:
        supervised_logger.info("Loading data for {}".format(setting))
        X=unpickle_obj("pickle_dir\\{}\\X_{}_pickle".format(setting.feature_selection,setting.feature_selection))
        ref_index=unpickle_obj("pickle_dir\\{}\\refIndex_{}_pickle".format(*itertools.repeat(setting.feature_selection,2)))
        y=unpickle_obj("pickle_dir\\{}\\y_{}_pickle".format(*itertools.repeat(setting.feature_selection,2)))
        y=np.array([list(set((normalize_genre_string(g,1) for g in g_list))) for g_list in y])

        #filter out unwanted genres
        X_filtered,y_filtered,ref_index_filtered=filter_genres(X,y,ref_index,ignore_genre)
        ref_indexes_unmatched.append(ref_index_filtered)
        Xs.append(X_filtered)
        ys.append(y_filtered)

    #match refids
    supervised_logger.info("Making ref indexes match for the data sets")
    Xs,ys,ref_indexes=match_sets_based_on_ref_id(Xs,ys,ref_indexes_unmatched)

    #make sure ref indexes match
    match=True
    prev_index=ref_indexes_unmatched[0]
    for ref_index in ref_indexes_unmatched[1:]:
        match=(prev_index==ref_index).all()
        prev_index=ref_index

    if not match:
        raise AttributeError("The matrices's reference indexes do not match, proceeding will resulting in wrong mapping"
                             "between instances")

    """
    INITIALIZE CLASSIFIERS
    """
    classifier=classifiers.ClassifierUtil()

    for setting in settings:
        threshold=setting.threshold
        ll_ranking=setting.ll_ranking
        setting.classifier_list=[#classifiers.Ada(threshold=threshold,ll_ranking=ll_ranking,base_estimator=MultinomialNB()),
                      #classifiers.kNN(n_neighbors=16,threshold=threshold,ll_ranking=ll_ranking),
                      #classifiers.RandomForest(threshold=threshold,ll_ranking=ll_ranking),
                      #classifiers.mNB(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.LogisticRegression(threshold=threshold,ll_ranking=ll_ranking),
                      #classifiers.DecisionTree(threshold=threshold,ll_ranking=ll_ranking),
                      #classifiers.SVC(probability=True,threshold=threshold,ll_ranking=ll_ranking)
                                 ]



    best_result=(0,("w1","w2"),"classifier_name",["num_attributes"])

    #This is just the cross product of different attribute counts.
    supervised_logger.info("We are using {} different attribute".format([len(setting.num_attributes) for setting in settings]))
    for num_attrs in itertools.product(*[setting.num_attributes for setting in settings]):


        num_attrs=list(num_attrs)

        cv_res=ResCrossValidation() #store crossvalidation results
        for fold,(train_index,test_index) in enumerate(randomized_cross_validation_gen(Xs[0].shape[0],global_settings.k_folds)):
            supervised_logger.info("On the {}th fold currently".format(fold))

            """
            LOAD TRAINING AND TESTING SETS WITH CROSS VALIDATION
            """

            train_sets,test_sets=load_training_testing(Xs,ys,ref_indexes,settings,train_index,test_index)

            train_y=train_sets[0].y
            train_ref_indexes=train_sets[0].ref_index

            test_y=test_sets[0].y
            test_ref_indexes=test_sets[0].ref_index

            """
            FEATURE SELECTION and EXTRACTION
            """

            feature_selector=feature_selection_strategy[global_settings.feature_selector_name]
            #feature_selector= PerClassFeatureSelector(*[SelectKBest(chi2,setting.num_attributes//num_genres)])

            train_Xs,test_Xs=feature_selection(settings,feature_selector,train_sets,test_sets,num_attrs)

            train_set=MultiData(train_Xs,train_y,train_ref_indexes)

            test_set=MultiData(test_Xs,test_y,test_ref_indexes)

            """
            CLASSIFICATION
            """
            classifier_name_to_accuracy=classify(classifier_util,settings,train_set,test_set,weights,global_settings.print_res)

            #Go through each classifier's ResultSingle for different settings
            for c_n, res_single_list in classifier_name_to_accuracy.items():
                cv_res.update(*res_single_list,kth_fold=fold)

        results_list=cv_res.results

        for res in results_list:
            print("Result: {}".format(tuple(res)+(num_attrs,)))

        best_result_at_curr_num_attr=tuple(max(results_list,key=op.itemgetter(0)))+(num_attrs,)

        #classifier_name_to_accracy is an association b/w a string of name of all the classifiers and the best weight for each
        supervised_logger.info("Best accuracy achieved at: {} with num features {}".format(
            best_result_at_curr_num_attr,num_attrs))

        best_result=max(best_result,best_result_at_curr_num_attr,key=op.itemgetter(0))

    print("Absolute best result at {}".format(best_result))




