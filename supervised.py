import itertools

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
from data.training_testing import Training,Testing,randomized_training_testing
from data import LearningSettings
from data.util import unpickle_obj,flatten_training
from util.base_util import normalize_genre_string
from classification_attribute.feature_selection import PerClassFeatureSelector


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

def filter_genres(X,y,ref_indexes):
    """
    Cleans up the label set, remove those in the ignore list from the websites.

    :param X:
    :param y:
    :param ref_indexes:
    :return:
    """
    removal_count={g:0 for g in ignore_genre}

    for index,g_list in enumerate(y):
        y[index]=list(set(g_list)-ignore_genre)

    keep_index=np.array([i!=[] for i in y])

    print("Eliminated {} webpages".format(y.shape[0]-sum(keep_index)))

    return X[keep_index],y[keep_index],ref_indexes[keep_index]


if __name__=="__main__":
    #test_set_nums,genre_dict=pick_random_test(genre_dict)
    #
    # map_urlAllGram(genre_dict)

    res_dir="classification_res"
    pickle_dir="pickle_dir"

    #CLASSIFICATION SETTINGS
    settings=LearningSettings(type="supervised",dim_reduction="chi_sq",num_feats=0,feature_selection="summary",
                              pickle_dir=pickle_dir,res_dir=res_dir)
    settings.result_file_label="no_region_kids_home_news"
    threshold=4
    ll_ranking=False
    num_attributes={10000}

    train_set_size=50000
    random_pick_test_training=False


    #LOAD AND PREPROCESS DATA SETS
    X=unpickle_obj("pickle_dir\\X_summary_pickle")
    y=unpickle_obj("pickle_dir\\y_summary_pickle")
    #normalize genres
    y=np.array([list(set((normalize_genre_string(g,1) for g in g_list))) for g_list in y])
    ref_indexes=unpickle_obj("pickle_dir\\refIndex_summary_pickle")

    #filter out unwanted genres
    X,y,ref_indexes=filter_genres(X,y,ref_indexes)


    #CLASSIFIERS
    classifier=classifiers.Classifier()
    classifiers_list=[classifiers.kNN(n_neighbors=16,threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.LogisticRegression(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.RandomForest(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.mNB(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.DecisionTree(threshold=threshold,ll_ranking=ll_ranking),
                      classifiers.SVC(probability=True,threshold=threshold,ll_ranking=ll_ranking)]



    for i in num_attributes:
        settings.num_feats=i

        #LOAD TRAINING AND TESTING
        #randomly pick from the entire set
        if random_pick_test_training:
            train_set,test_set=randomized_training_testing(settings,X,y,ref_indexes,train_set_size)
        else:
            train_set=Training(settings,pickle_dir=settings.pickle_dir)
            train_set.load_training(secondary_label=settings.result_file_label)

            test_set=Testing(settings,pickle_dir=settings.pickle_dir)
            test_set.load_testing(secondary_label=settings.result_file_label)

        #FEATURE SELECTION,FLATTEN TRAINIGN
        #count number of classes there are
        num_genres=len(set(itertools.chain(*([i for i in i_list]for i_list in train_set.y))))
        feature_selector=SelectKBest(chi2,i)
        #PerClassFeatureSelector(*[SelectKBest(chi2,i//num_genres)])

        #flatten training
        flatten_training(train_set)

        feature_selection(train_set,test_set,feature_selector,fit=True)

        #CLASSIFICATION
        classifier=classifiers.Classifier()
        classifier.classify(settings,train_set,test_set,classifiers_list)

