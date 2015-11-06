from db.db_model.mongo_websites_models import *

__author__ = 'Kevin'

import db.db_collections.mongo_collections as coll
from util.base_util import *

"""
utilities that focuses on evolving the program

"""

'''
Resolve issues with bad parents

'''
def print_url_and_parents(file):
    with open(file,encoding='iso-8859-1',mode='a') as output:
        for url_obj in coll.URLToGenre().find():
            try:
                output.write(unreplace_dot_url(url_obj['url']+'\n'))

                if 'parent' in url_obj:
                    for p in url_obj['parent']:
                        output.write('\t\t:::{}\n'.format(unreplace_dot_url(p.url)))
            except:
                print('url error {}'.format(url_obj['id']))


def genre_to_genre_data(url_obj):
    """
    Used to reverse the bad decision I made about having a direct connection from URLToGenre table to Genre table.
     in GenreMetaData table
    This method will make a new entry in genremeta data and shift all genres to that table

    :param url_obj:
    :return:
    """
    assert isinstance(url_obj,URLToGenre)

    if not hasattr(url_obj,'genre_data') or len(coll.GenreMetaData().select(url=url_obj['url']).find()) is 0 or \
        len(url_obj['genre_data']['genres'])==0:
        genres=[]
        for genre in url_obj['genre']:
            genres.append(EmbeddedGenre(type='alexa',count=1,genre=genre))

        if not hasattr(url_obj,"genre_data"):
            genre_meta_obj=GenreMetaData(url=url_obj['url'],genres=genres)
            genre_meta_obj.save()
        else:
            GenreMetaData.objects(url=url_obj["url"]).update(genres=genres)
            genre_meta_obj=GenreMetaData.objects(url=url_obj["url"])[0]

        URLToGenre.objects(url=url_obj['url']).update_one(genre_data=genre_meta_obj)
        url_obj=url_obj.reload()


    return url_obj