__author__ = 'Kevin'
from util.base_util import *

from db.db_model.mongo_websites_models import Genre,EmbeddedGenre,GenreTest,GenreMetaData,GenreMetaDataTest
from mongoengine import DoesNotExist

import collections as coll

"""
Define special operations on mongodb collection provided in addition to the basic MongoEngine operations.
"""


class Genres:

    @staticmethod
    def create_genres(genre_list,url):
        """
        Create entries in the genre table from a list of string genre. This method will normalize all genre with
        base_util's normalize genre method

        If genre exists, return it instead of creating a new one.
        Also, this method makes sure that there are no repeats

        :param genre_list: Each genre level should be delimited by / and be of type str
        :return: List of Genre collection references to documents.
        """

        assert isinstance(genre_list,list)
        genre_list=set(normalize_genre_string(gs) for gs in genre_list)

        genre_refs=list()
        for genre_string in genre_list:
            assert isinstance(genre_string,str)

            try:
                genre_ref=GenreTest.objects.get(genre=genre_string)

                genre_refs.append(genre_ref)
            except DoesNotExist:
                genre_ref=GenreTest.objects(genre=genre_string).modify(set__urls=[url],upsert=True,new=True)
                genre_refs.append(genre_ref)

        return genre_refs

#embedded document in each document in GenreMetaData
class GenreMetaData:

    @staticmethod
    def create_genremetadata(genre_ref_list,url):
        """
        Create genre metadata for url if it does not exist already and insert the list.

        Avoids repeat of same genre to genre_ref_list.

        :param genre_ref_list: A list of EmbeddedGenreEntry. Cannot be empty and should not have repeats
        :return:
        """
        assert url is not None
        assert genre_ref_list is not []
        assert isinstance(genre_ref_list,list)

        try:
            genre_metadata_ref=GenreMetaDataTest.objects.get(url=url)
            genre_string_set=set(embedded_genre.genre.genre for embedded_genre in genre_metadata_ref.genres)

            new_embedded_genre_list=[]
            #get all the genres not in the genre metadata
            for embedded_genre_entry in genre_ref_list:


                if embedded_genre_entry.genre.genre not in genre_string_set:
                    new_embedded_genre_list.append(embedded_genre_entry)

            #finally update
            genre_metadata_ref.update(push_all__genres=new_embedded_genre_list)
            genre_metadata_ref.reload()

        except DoesNotExist:
            GenreMetaDataTest(url=url,genres=genre_ref_list).save()
            genre_metadata_ref=GenreMetaDataTest.objects.get(url=url)

        return genre_metadata_ref

