__author__ = 'Kevin'

from .mongo_collection_base import MongoCollection
import db.db_model.mongo_websites_models as models
import warnings

'''
This module contains collection of ORM-like objects that serves as an easy interface for each
    collection in mongodb.

'''



class Analytics(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.Analytics)

        if len(kwargs)>0:
            self.create(**kwargs)

class Genre(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.Genre)


class GenreMetaData(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.GenreMetaData)

    """
    create embedded genres from passed in genre objects with default value of alexa type and NA result type
    """
    def create_embedded_genres(self,*genres,count=0,type='alexa',result_type='NA'):
        return [models.EmbeddedGenre(type=type,genre=g,count=count,result_type=result_type) for g in genres]

class MetaData(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.MetaData,**kwargs)

    def get_increment_position(self,queue_name,increment=1):
        """
        Get the position in queue queue_name and increment to the next position

        :param queue_name:
        :return:
        """
        pos=self.select(type=queue_name).find_one()['position']

        self.select(type=queue_name).update(position=pos+increment)

        return pos

class QueryResults(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.QueryResults)

class URLToGenre(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.URLToGenre)

    def bow_contains_word(self,url,w):
        #check if bow exists first
        res=self.select(url=url).find_one(only="bow")

        return w in res if len(res)>0 else None

class URLQueue(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.URLQueue)

    def create_queue_from_iterable(self,url_iterable):
        for c,url_doc in enumerate(url_iterable):
            self.create(document=url_doc,number=c)

    #use in conjunction with MetaData Table
    def iter(self,queue_name):

        res=self.select(number=MetaData().get_increment_position(queue_name)).find_one()

        acc=0
        while res != None and acc<5:
            yield res
            res=self.select(number=MetaData().get_increment_position(queue_name)).find_one()

            if res==None:
                acc+=1
            else:
                acc=0


        raise StopIteration('End of Queue')

class CategoricalBOW(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.CategoricalBOW)

    def update_genre_word_count(self,genre,word_dict):
        """
        Update the genre word count. Create the genre if it does not exist and add in the word along with incrementing it
        :param genre:
        :param word_dict:
        :param increment:
        :return:
        """
        error_terms=open('error_terms',mode='a')

        genre_bow_obj=self.select(genre=genre).find_one()

        #if genre none exist create!
        if genre_bow_obj is None:
            self.create(genre=genre)
            genre_bow_obj=self.select(genre=genre).find_one()

        # see if word exists, if it does not exist, it should now have a count from what we counted before
        for k in word_dict.keys():
            word_dict[k]+=genre_bow_obj.bow.get(k,0)

        #the update query
        update_query=dict(("bow__{}".format(k),v) for k,v in word_dict.items())


        #we need custom query to update the inner dictionary of words
        for k,v in update_query.items():
            try:
                self.select(genre=genre).update(**{k:v})
            except:
                error_terms.write("{}:::{}, {}\n".format(genre,k,v))

class ShortGenre(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.ShortGenre)

class URLBow(MongoCollection):
    """
    Collection that maps URL to bow.

    This is done so that we don't take up too much space in URLToGenre and degrade its speed
    """
    def __init__(self,**kwargs):
        super().__init__(models.URLBow)

class MutualInformation(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.MutualInformation)

class Top30WordGenre(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.Top30WordGenre)

class WordGenreJoint(MongoCollection):
    def __init__(self,**kwargs):
        super().__init__(models.WordGenreJoint)


