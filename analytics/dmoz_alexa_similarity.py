__author__ = 'Kevin'

import sys

from db.db_model.mongo_websites_models import *
from util.base_util import *

'''
For the urls in alexa that has a corresponding in dmoz.

Calculate how many

'''
def dmoz_alexa_similarity():
    #our results
    similarity_res={}

    #make sure that there are no repeats of urls
    urls=set()

    for genre_meta in GenreMetaData.objects:
        if genre_meta['url'] not in urls:
            urls.add(genre_meta['url'])
            #Logger.info('Looking at url {}'.format(genre_meta['url']))

            alexa_bucket,dmoz_bucket=_extract_genre_on_type(genre_meta['genres'])

            #if dmoz bucket is not empty, then this url has some matches in dmoz database
            if len(dmoz_bucket) >0:
                similarity_res['dmoz_url_total']=similarity_res.get('dmoz_url_total',0)+1

            analytics_obj=_calculate_similarity(alexa_bucket,dmoz_bucket,genre_meta['url'])

            for k in analytics_obj.keys():
                similarity_res[k]=analytics_obj[k]+similarity_res.get(k,0)

    print(similarity_res)

'''
Extracts each url's dmoz and alexa genre metadata objects
'''
def _extract_genre_on_type(genre_info_list):
    alexa_bucket,dmoz_bucket=[],[]

    for genre_info in genre_info_list:
        if genre_info['type'] =='alexa':
            alexa_bucket.append(genre_info)

        elif hasattr(genre_info,'result_type') or 'result_type' in genre_info:

            if genre_info['result_type']=='url':

                dmoz_bucket.append(genre_info)

    return alexa_bucket,dmoz_bucket
'''
Edit distance is only for urls that do not have
'''
def _calculate_similarity(alexa_genres,dmoz_genres,url):
    alexa_dict=dict(zip((normalize_genre_string(g['genre']['genre'],2) for g in alexa_genres),alexa_genres))
    dmoz_dict=dict(zip((normalize_genre_string(g['genre']['genre'],2) for g in dmoz_genres),dmoz_genres))

    genre_exact_match=0
    edit_distance_count=0
    total_edit_distance=0
    dmoz_genre_length=0
    dmoz_genre_num=0
    edit_distance=(sys.maxsize,'')
    total_genre_pairs=0
    #find exact matchs with alexa; if not a match, calculate the edit distance
    for dmoz_genre_name,dmoz_genre_info in dmoz_dict.items():
        curr_edit_distance=(sys.maxsize,'')
        dmoz_genre_num+=1

        dmoz_genre_length+=len(dmoz_genre_name)
        for alexa_genre_name,alexa_genre_info in alexa_dict.items():
            if edit_distance is 0:
                break

            if alexa_genre_name == dmoz_genre_name:
                genre_exact_match+=1
                curr_edit_distance=(0,alexa_genre_name)

            else:
                curr_edit_distance=min(curr_edit_distance,(levenshtein(dmoz_genre_name,alexa_genre_name),alexa_genre_name),key=lambda x: x[0])

        edit_distance=min(edit_distance,curr_edit_distance,key=lambda x: x[0])

    if edit_distance[0] is not sys.maxsize and edit_distance[0] is not 0:
        print('{}:::{}:::{}'.format(unreplace_dot_url(url),dmoz_genre_name,edit_distance[1]))
        total_edit_distance+=edit_distance[0]
        edit_distance_count+=1

    return {'dmoz_url_match':1 if genre_exact_match>0 else 0,'edit_distance_count':edit_distance_count,
            'total_edit_distance':total_edit_distance,'dmoz_genre_length':dmoz_genre_length,'num_dmoz_genre':dmoz_genre_num
            ,'exact_genre_matches':genre_exact_match,'total_genre_pairs':total_genre_pairs}




