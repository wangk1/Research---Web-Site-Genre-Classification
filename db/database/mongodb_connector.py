__author__ = 'Kevin'

import warnings

from db.db_model.mongo_websites_models import *
from util.Logger import Logger
from util.exception_annotations import fail_safe_encode,fail_safe_mongo
from util.base_util import *

warnings.warn('Mongodb Connector Class is now deprecated as a means of accessing the Mongo DB')

queue_types=['url_queue','url_active_queue']

"""
Main class for transactions with Mongo Database

Has alot of helper methods
"""
class MongoDB:
    #self.mongo

    def __init__(self,host,port):
        #tell mongoengine to connect
        self.connect(host=host,port=port)

    @classmethod
    def connect(cls,host,port):
        connect(db=settings.USE_DB,host=host,port=port)

        cls.setup()
        return cls

    @classmethod
    def setup(cls):
        if not MetaData.objects(type='queue'):
            MetaData(type='queue').save()

    """
    Save/Modify a url into the URLTo along with its genres.

    If url exists, modify the url's genres.

    If url does not exist, create new url entry in URLToGenre collection.

    If genre does not exist, this method will delegate to get_genre_refs to create the nonexistent
        genre's entry in Genre document. The returned list of genre document reference

    Exceptions: This method is fail_safe see @fail_safe

    """
    @staticmethod
    @fail_safe_mongo
    def save_modify_url(**kwargs):
        kwargs['url']=replace_dot_url(kwargs['url'])

        #optionally turn off genre parsing, use for if the genre passed in are references instead of string
        if 'genre_no_parse' in kwargs and not kwargs['genre_no_parse']:
            del kwargs['genre_no_parse']

        else:
            kwargs['genre']=MongoDB.get_genre_refs(*kwargs['genre'])

        url_to_genre_resultset=URLToGenre.objects(url=kwargs['url'])

        #if url does not exist, create a new url entry in document URLToGenre
        #add in the respective genres
        if url_to_genre_resultset.count() ==0:
            MongoDB.save_url(**kwargs)

        else:
            #only first url from result set
            kwargs['url']=url_to_genre_resultset[0]

            MongoDB.modify_url(url_model_provided=True,**kwargs)



    @staticmethod
    def save_url(**kwargs):
        kwargs['url']=replace_dot_url(kwargs['url'])

        url_model=URLToGenre(**kwargs)

        try:
            save_obj=url_model.save()
        except:
            Logger.error("Error saving: "+str(kwargs['url']))

        return save_obj


    """

    """
    @staticmethod
    def modify_url(**kwargs):
        if 'url' not in kwargs:
            raise IndexError("Mongo Connector: URL NOT EXISTS in call to Modify URL")

        url_model_provided=pop_attr('url_model_provided',kwargs) if 'url_model_provided' in kwargs else False

        url_model=kwargs.pop('url')

        #just a string url instead of mongo db url document
        if not url_model_provided:
            url_model=replace_dot_url(url_model)

            url_model=URLToGenre.objects(url=url_model)[0]

        update_query={}
        #for each field to be updated
        for (k,v) in kwargs.items():
            if k in url_model and isinstance(url_model[k],list):
                    update_query['push_all__'+k]=v

            else:
                update_query[k]=v

        #finally do the update
        update_obj=URLToGenre.objects(url=url_model['url']).update(**update_query)

        return update_obj

    """
    Convert dictionary with genre information to actual mongodb document object for the genre if exists.

    If the genre document does not exist, create a new one from dictionary information.

    """
    @staticmethod
    @fail_safe_mongo
    def get_genre_refs(*genres):

        genre_objs=[]
        for agenre_obj in genres:
            #find all matching genre
            if isinstance(agenre_obj,str):
                genre_models=Genre.objects(genre=agenre_obj)
            else:
                genre_models=Genre.objects(genre=agenre_obj['genre'])

            if(len(genre_models)==0):
                if not isinstance(agenre_obj,str):
                    genre_model=Genre()

                    for (k,v) in agenre_obj.items():
                        genre_model[k]=v
                else:
                    genre_model=Genre(genre=agenre_obj)

                try:
                    genre_model.save()
                except:
                    Logger.error("Error saving: "+str(agenre_obj))


                genre_objs.append(genre_model)
            else:
                genre_objs.extend(genre_models.all())

        return genre_objs


    @staticmethod
    def get_genre(url):
        url=replace_dot_url(url)

        genres=URLToGenre.objects(url=url)
        return None if len(genres)==0 else genres[0]

    @staticmethod
    def update_url(url,new_url):
        url=replace_dot_url(url)
        new_url=replace_dot_url(new_url)

        Logger.info('Updating {} to {}'.format(url,new_url))
        return URLToGenre.objects(url=url).update(url=new_url)


    @staticmethod
    def get_url_object(url):
        url=replace_dot_url(url)

        match=URLToGenre.objects(url=url)

        return match[0] if len(match) >0 else None

    @staticmethod
    @fail_safe_mongo
    def save_page(url,page):
        url=replace_dot_url(url)

        url_objects=URLToGenre.objects(url=url)

        return url_objects.update(page=page)


    @staticmethod
    @fail_safe_encode
    def push_to_queue(number,url_doc):
        try:

            if URLQueue.objects(number=number):
                return None


            URLQueue(number=number,document=url_doc).save()
        except:
            try:
                Logger.error('Failed to save with url: {}'.format(url_doc['url']))
            except:
                Logger.error('Complete error to save number: {}'+number)


    @staticmethod
    @fail_safe_mongo
    def get(doc,field,**criteria):
        obj_col=doc.iterable(**criteria)

        if len(obj_col) is not 0:
            return obj_col[0][field]
        return None

    @staticmethod
    def get_m(doc,field,**criteria):
        obj_col=doc.iterable(**criteria)

        if len(obj_col) is not 0:
            return [obj[field] for obj in obj_col]
        return []

    @staticmethod
    def increment_url_counter(amt=1):
        count=MetaData.objects(type='queue')[0].position+amt

        MetaData.objects(type='queue').update(position=count)

        return count

    @staticmethod
    def count(collect):
        return collect.iterable.count()