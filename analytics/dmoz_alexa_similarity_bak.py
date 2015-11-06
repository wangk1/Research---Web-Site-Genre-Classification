__author__ = 'Kevin'

"""
Package for calculating the similarity between DMOZ and alexa ranking

Must connect to db first
"""

import sys

from db.database import DBQueue
from db.database import MongoDB
from db.db_model.mongo_websites_models import *
from util.base_util import *
import db_collections.mongo_collections as col
from util.Logger import Logger

ANALYTICS_NAME='alexa_dmoz_comp'

def calculate_similarity():
    q=DBQueue('similarity_queue')
    genre_meta_data=GenreMetaData.objects.order_by("url")[q.get():]

    #init the Analytics
    analytics_coll=col.Analytics()

    if analytics_coll.select(name=ANALYTICS_NAME).find_one() is None:
        analytics_coll.create(alexa_total=0,edit_distance_count=0, total_edit_distance=0,alexa_match=0,name=ANALYTICS_NAME,
                              alexa_genre_length=0)

    urls=set()
    #calculate the similar on a document to document basis
    for genre_meta in genre_meta_data:

        if genre_meta['url'] not in urls:
            urls.add(genre_meta['url'])

            Logger.info('Doing genre for url: {}'.format(genre_meta['url']))

            similarity_res=_calculate_similarity_document(genre_meta)

            analytics_obj=analytics_coll.select(name=ANALYTICS_NAME).find_one()

            for k in similarity_res.keys():
                similarity_res[k]+=analytics_obj[k]

            analytics_coll.select(name=ANALYTICS_NAME).update(**similarity_res)
            q.increment()


    print('URL has a unique percent of {}'.format(len(urls)/len(genre_meta_data)*100))

"""
Calculate each document's similarity
"""
def _calculate_similarity_document(genremeta_obj):
    alexa_genres,dmoz_genres=_extract_genre_on_type(genremeta_obj['genres'])

    similar_res=calculate_alexa_dmoz_similarity(alexa_genres,dmoz_genres)

    return similar_res

def _extract_genre_on_type(genre_info_list):
    alexa_bucket,dmoz_bucket=[],[]

    for genre_info in genre_info_list:
        if genre_info['type'] =='alexa':
            alexa_bucket.append(genre_info)

        else:
            dmoz_bucket.append(genre_info)

    return alexa_bucket,dmoz_bucket

"""
Single document's alexa and dmoz similarity

"""
def calculate_alexa_dmoz_similarity(alexa_genres,dmoz_genres):
    alexa_dict=dict(zip((normalize_genre_string(g['genre']['genre'],2) for g in alexa_genres),alexa_genres))
    dmoz_dict=dict(zip((normalize_genre_string(g['genre']['genre'],2) for g in dmoz_genres),dmoz_genres))

    exact_match=0
    edit_distance_count=0
    total_edit_distance=0
    alexa_genre_length=0
    #find exact matchs with alexa; if not a match, calculate the edit distance
    for alexa_genre_name,alexa_genre_info in alexa_dict.items():
        edit_distance=sys.maxsize

        alexa_genre_length+=len(alexa_genre_name)
        for dmoz_genre_name,dmoz_genre_info in dmoz_dict.items():
            if edit_distance is 0:
                break

            if alexa_genre_name == dmoz_genre_name:
                exact_match+=1
                edit_distance=0

            else:
                edit_distance=min(edit_distance,levenshtein(dmoz_genre_name,alexa_genre_name))

        if edit_distance is not sys.maxsize:
            total_edit_distance+=edit_distance
            edit_distance_count+=1

    return {'alexa_total':len(alexa_dict),'edit_distance_count':edit_distance_count, 'total_edit_distance':total_edit_distance,'alexa_match':exact_match,
            'alexa_genre_length':alexa_genre_length}



    #accumulate stats and update the collection


MongoDB.connect(settings.HOST_NAME,settings.PORT)

