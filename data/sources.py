import itertools

__author__ = 'Kevin'


import random
import util.base_util as util
from sklearn.feature_selection import chi2 as chi_sq,SelectKBest
from sklearn.pipeline import Pipeline

from collections import namedtuple
from db.db_model.mongo_websites_classification import URLAllGram,TestSet_urlAllGram,TrainSet_urlAllGram,URLBow_fulltxt, \
    TrainSet_urlFullTextBow,TestSet_urlFullTextBow
from db.db_model.mongo_websites_models import TestSetBow,TrainSetBow
import classification.classifiers as classifiers
import math

ClassificationSource=namedtuple("Source",("ref_index","attr_map","short_genre"))
def generate_training_testing(source,test_set_indexes,genres,*,train_coll_cls,test_coll_cls):
    """
    Takes a list of test set indexes and genres and save to their respective collections via
        aka train_coll_cls and test_coll_cls

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

class SourceMapper:
    """
    Maps a source iterable's attribute fields to another iterable containing the now
        renamed attributes

    This is a special cached generator. It can give an infinite copies of the generator
        that walks over the source.

    """

    def __init__(self,source,mapping):
        assert len(mapping)>0

        self.source=source
        self.mapping=mapping


    def __iter__(self):
        return self.iter()

    def iter(self):
        mapped_obj={}

        for src_obj in self.source:
            for k,v in self.mapping.items():
                mapped_obj[v]=src_obj[k]

            yield ClassificationSource(**mapped_obj)

        raise StopIteration("End of source")