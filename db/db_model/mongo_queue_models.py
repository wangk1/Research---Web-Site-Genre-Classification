__author__ = 'Kevin'
from mongoengine import *
from .mongo_websites_models import URLToGenre

class Queue_full_page(DynamicDocument):

    url=StringField(required=True)
    number=IntField(required=True)

    meta={'collection':'Queue_full_page',
          'indexes':[
              'number'

          ],
          "db_alias":"Websites_queue"}

class QueueTracker(Document):
    """
    Track the location we are currently at in the queue

    """

    name=StringField(required=True)
    location=IntField(required=True,default=0)

    meta={'collection':'QueueTracker',
          'indexes':[
              'name'

          ],
          "db_alias":"Websites_queue"}