__author__ = 'Kevin'
from .db_model.mongo_queue_models import QueueTracker

from db.db_collections.mongo_collections import MetaData
from util.Logger import Logger
from mongoengine import connection, OperationError, ValidationError
import settings as global_settings
import db.settings as db_settings

"""
This init file will contain the DBQueue class and the connectors for the databases in mongo db
"""

class DBQueue:
    #self.queue_cls
    #self.queue_name

    def __init__(self,queue_cls,queue_name):
        self.queue_name=queue_name
        self.queue_cls=queue_cls

        #new queue, we must create new entry
        if len(QueueTracker.objects(name=queue_name))==0:
           QueueTracker(name=queue_name).save()

    def _get_tracker_document(self):
        return QueueTracker.objects.get(name=self.queue_name)

    def increment_location(self,count=1):
        self._get_tracker_document().update(inc__location=count)


    def decrement_location(self,count=1):
        self._get_tracker_document().update(dec__location=count)

    def set_location(self,count):
        self._get_tracker_document().update(set__location=count)

    def get_location(self):
        return self._get_tracker_document().location

    def create_queue(self,iterable):
        """
        Populate the queue from each item in the iterable where the reference of each item is the document of the queue.
        """

        for c,doc_obj in enumerate(filter(lambda x: x, iterable)):
            c%1000==0 and Logger.info("{} done".format(c))


            self.queue_cls(number=c,url=doc_obj.url).save()



class DBQueue_old:
    """
    A queue in the database with the specified type(Just a unique identifier).

    This is used so that progress can be saved to database

    """
    #self.type
    #self.queue

    def __init__(self,type_queue,position=0):
        self.queue=MetaData(type=type_queue)

        if self.queue.find_one()==None:
            Logger.info('Queue of Type: {} does not exist in database, creating'.format(type_queue))
            self.queue.create(type=type_queue,position=position).save()


    def __get_queue(self):
        return self.queue.find_one()

    def increment(self,interval=1):
        queue=self.__get_queue()
        new_pos=queue.position+interval

        self.queue.update(position=new_pos)

        return new_pos


    def decrement(self,interval=1):
        pos=self.queue.find_one().position
        new_pos=pos-interval

        self.queue.update(position=new_pos)

        return new_pos

    def set(self,count=0):
        self.queue.update(position=count)

    def get(self):
        return self.queue.find_one().position

#Websites db
connection.connect(db=global_settings.USE_DB,host=global_settings.HOST_NAME,port=global_settings.PORT)
connection.connect(db=db_settings.MUTUAL_INFO_DB,host=global_settings.HOST_NAME,port=global_settings.PORT,alias=db_settings.MUTUAL_INFO_DB)
connection.connect(db=db_settings.CLASSIFICATION_DB,host=global_settings.HOST_NAME,port=global_settings.PORT,alias=db_settings.CLASSIFICATION_DB)
connection.connect(db=db_settings.QUEUE_DB,host=global_settings.HOST_NAME,port=global_settings.PORT,alias=db_settings.QUEUE_DB)
