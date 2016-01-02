import functools
import itertools
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

__author__ = 'Kevin'
import util.base_util as util,numpy as np
from classification_attribute.feature_selection import feature_selection

from collections import namedtuple
from db.db_model.mongo_websites_classification import URLAllGram,TestSet_urlAllGram,TrainSet_urlAllGram,URLBow_fulltxt, \
    TrainSet_urlFullTextBow,TestSet_urlFullTextBow
import classification.classifiers as classifiers
from data.training_testing import Training,Testing,randomized_training_testing_sets,MultiData
from data import LearningSettings
from data.util import unpickle_obj,flatten_training
from util.base_util import normalize_genre_string
from classification_attribute.feature_selection import PerClassFeatureSelector
from util.genre import filter_genres
from util.Logger import Logger
from classification.classifiers import MultiClassifier
from data.X_y import match_sets_based_on_ref_id

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

def get_all_gram_genres():
    genres_iter=(set(util.normalize_genre_string(g,1) for g in allgram_obj.short_genres) for allgram_obj in URLAllGram.objects)

    return itertools.chain(*genres_iter)

def get_full_text_genres():
    genres_iter=(set(util.normalize_genre_string(g,1) for g in url_bow_obj.short_genres) for url_bow_obj in URLBow_fulltxt.objects)

    return itertools.chain(*genres_iter)

def map_urlAllGram(genre_dict):

    mapped_obj=(ClassificationSource(url_allgram_obj.ngram_index,url_allgram_obj.ngram,
                                                    util.normalize_genre_string(url_allgram_obj.short_genres[0],1))
                                    for url_allgram_obj in URLAllGram.objects)

    generate_training_testing( mapped_obj
                              ,test_set_nums,genre_dict.keys()
                              ,train_coll_cls=TrainSet_urlAllGram,test_coll_cls=TestSet_urlAllGram)

def map_urlFullText(genre_dict):


    generate_training_testing((ClassificationSource(url_fulltxt_obj.bow_index,url_fulltxt_obj.bow,
                                                    util.normalize_genre_string(url_fulltxt_obj.short_genres[0],1))
                                    for url_fulltxt_obj in URLBow_fulltxt.objects)
                              ,test_set_nums,genre_dict.keys()
                              ,train_coll_cls=TrainSet_urlFullTextBow,test_coll_cls=TestSet_urlFullTextBow)

#genres to filter out with filter genre methods
ignore_genre={
    "World",
    "id",
    "desc",
    "page",
    "parent",
    "url",
    "genres",
    "Kids_and_Teens",
    "Kids",
    "Regional",
    "Home",
    "News"
}



def classify(classifier_util,learn_settings,train_set,test_set,classifier,increment=500):
        """
        Classify with test_X and generate result files for each classifier set

        :param classifiers: set of additional classifier to test with in addition to self.classifier classifiers
        :return:
        """


        classifier_name=str(classifier)


        supervised_logger.info("Classifying with {}".format(classifier_name))
        classifier.fit(train_set.X,train_set.y)

        supervised_logger.info("Fitting done, predicting with test set")

        res=classifier\
                .predict_multi(test_set.X)

        print("Done, printing Results for {}".format(classifier_name))
        classifier_util.print_res(learn_settings,
              y=test_set.y,
              predictions=res,
              ref_indexes=test_set.ref_indexes,
              classifier_name=classifier_name)

def load_training_testing(Xs,y,ref_index,settings,train_set_size,random_pick_test_training):
    """
    Load training and testing set or randomly pick then from Xs,y,and ref_index.

    :param Xs:
    :param y:
    :param ref_index:
    :param settings:
    :param train_set_size:
    :return: List of train_set and test_set objs
    """
    if not (hasattr(y,"shape") and hasattr(ref_index,"shape")):
        raise AttributeError("Only 1 set of common y(label) and ref index accepted for all Xs")

    if random_pick_test_training:
        train_sets,test_sets=randomized_training_testing_sets(settings,Xs,y,ref_index,train_set_size)
    else:
        train_sets=[]
        test_sets=[]

        for setting in settings:
            train_set=Training(setting,pickle_dir=setting.pickle_dir)
            train_set.load_training(secondary_label=setting.result_file_label)

            test_set=Testing(setting,pickle_dir=setting.pickle_dir)
            test_set.load_testing(secondary_label=setting.result_file_label)

            train_sets.append(train_set)
            test_sets.append(test_set)

    #flatten training
    for train_set in train_sets:
        flatten_training(train_set)

    return train_sets,test_sets


if __name__=="__main__":
    #test_set_nums,genre_dict=pick_random_test(genre_dict)
    #
    # map_urlAllGram(genre_dict)

    res_dir="classification_res"
    pickle_dir="pickle_dir"

    """
    CLASSIFICATION SETTINGS
    """
    setting=LearningSettings(type="supervised",dim_reduction="chi_sq",num_attributes=0,feature_selection="summary",
                              pickle_dir=pickle_dir,res_dir=res_dir)

    setting2=LearningSettings(type="supervised",dim_reduction="chi_sq",num_attributes=0,feature_selection="url",
                             pickle_dir=pickle_dir,res_dir=res_dir)
    settings=[setting,
              setting2
              ]

    for setting in settings:
        setting.result_file_label="no_region_kids_home_news"
        setting.threshold=4
        setting.ll_ranking=False
        setting.num_attributes=10000

    #GLOBAL SETTINGS
    train_set_size=50000
    random_pick_test_training=False

    """
    LOAD DATA, preprocess
    """
    #WARNING: REF INDEX for each individual X set must match row to row
    Xs=[]
    ys=[]
    ref_indexes_unmatched=[]

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
    TRAINING AND TESTING SETS LOADING, NO FEATURE EXTRACT and DIMENSIONALITY REDUCTION YET


    """
    supervised_logger.info("Generating or loading training samples")
    train_sets,test_sets=load_training_testing(Xs,ys[0],ref_indexes[0],settings,train_set_size,random_pick_test_training)


    """
    INITIALIZE CLASSIFIERS
    """
    classifier=classifiers.Classifier()

    for setting in settings:
        threshold=setting.threshold
        ll_ranking=setting.ll_ranking
        setting.classifier_list=[#classifiers.Ada(threshold=threshold,ll_ranking=ll_ranking,base_estimator=MultinomialNB()),
                      classifiers.kNN(n_neighbors=16,threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.LogisticRegression(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.RandomForest(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.mNB(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.DecisionTree(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.SVC(probability=True,threshold=threshold,ll_ranking=ll_ranking)]



    """
    FEATURE SELECTION and EXTRACTION
    """


    for index,setting in enumerate(settings):
        num_genres=len(set(itertools.chain(*([i for i in i_list]for i_list in train_sets[index].y))))

        feature_selector=SelectKBest(chi2,setting.num_attributes)
        #feature_selector= PerClassFeatureSelector(*[SelectKBest(chi2,setting.num_attributes//num_genres)])

        train_set=train_sets[index]
        test_set=test_sets[index]

        supervised_logger.info("Currently doing feature selection on {}th data set".format(index))
        train_set.X,test_set.X=feature_selection(train_set,test_set,feature_selector,fit=True)

    train_Xs=[train_set.X for train_set in train_sets]
    train_y=train_sets[0].y
    train_ref_indexes=train_sets[0].ref_indexes

    train_set=MultiData(train_Xs,train_y,train_ref_indexes)

    test_Xs=[test_set.X for test_set in test_sets]
    test_y=test_sets[0].y
    test_ref_indexes=test_sets[0].ref_indexes

    test_set=MultiData(test_Xs,test_y,test_ref_indexes)

    """
    CLASSIFICATION
    """

    for classifiers_list in itertools.product(*[setting.classifier_list for setting in settings]):
        multi_classifier=MultiClassifier(classifiers_list,threshold=1,ll_ranking=False)

        #CLASSIFICATION
        classifier_util=classifiers.Classifier()
        classify(classifier_util,settings,train_set,test_set,multi_classifier)



