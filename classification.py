import itertools

__author__ = 'Kevin'
import random
import util.base_util as util

from collections import namedtuple,Counter
from db.db_model.mongo_websites_classification import URLAllGram,TestSet_urlAllGram,TrainSet_urlAllGram,URLBow_fulltxt, \
    TrainSet_urlFullTextBow,TestSet_urlFullTextBow
from db.db_model.mongo_websites_models import TestSetBow,TrainSetBow
from classification.classifiers_func import classify,load_vocab_vectorizer
from classification.mapper import ClassificationSourceMapper
import math
from classification.classification_results import count_miss_ratio

ignore_genre={
    "World",
    "id",
    "desc",
    "page",
    "parent",
    "url",
    "genres",
    "Kids_and_Teens"
}

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


def pick_random_test(counter,percent_of_sample=0.1):
    """
    Randomly pick x percent of sample

    WARNING:be careful of repeating genres for the same element, this will throw the random pick off

    :param: genre_iterable: Iterable that produces genres to be counted
    :return: test picks, the nth items from each class we are choosing for the test set
    """
    # counter=Counter(util.normalize_genre_string(genre,1) for genre in genre_iterable
    #                 if util.normalize_genre_string(genre,1) not in ignore_genre)

    print(counter)

    test_picks={}
    for k,c in counter.items():
        #genre a few random numbers from 0 to number of urls in the genre. These random number are the
        #picks we will take for the testing set when we are iteration over the entire sample collection
        test_picks[k]=set(random.sample(range(0,c),math.floor(percent_of_sample*c)))
    return test_picks,counter

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

if __name__=="__main__":
    # test_set_nums,genre_dict=pick_random_test(genre_dict)
    #
    # map_urlAllGram(genre_dict)

    # load_vocab_vectorizer(TrainSetBow,extra_label="summary")
    # classify(train_coll_cls=TrainSetBow,test_coll_cls=TestSetBow,pickle_label="summary",k=100)

    # load_vocab_vectorizer(TrainSet_urlFullTextBow,"fulltxt")
    #classify(train_coll_cls=TrainSet_urlFullTextBow,test_coll_cls=TestSet_urlFullTextBow,pickle_label="fulltxt",k=200)

    # load_vocab_vectorizer(TrainSet_urlAllGram,"allgram")
    # classify(train_coll_cls=TrainSet_urlAllGram,test_coll_cls=TestSet_urlAllGram,pickle_label="allgram",k=200)

    #
    # generate_training_testing(test_set_num) np.vstack((np.array([1,2,3])

    count_miss_ratio()
