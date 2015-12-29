__author__ = 'Kevin'
from mongoengine import *

class TrainSet_urlFullTextBow(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TrainSet_urlFullTextBow',
          'indexes':[
              "ref_index"

          ],
          "db_alias":"Websites_classification"
          }

class TestSet_urlFullTextBow(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TestSet_urlFullTextBow',
          'indexes':[
              "ref_index"

          ],
          "db_alias":"Websites_classification"
          }

class TrainSet_urlAllGram(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TrainSet_urlAllGram',
          'indexes':[
              "ref_index"

          ],
          "db_alias":"Websites_classification"
          }

class TestSet_urlAllGram(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TestSet_urlAllGram',
          'indexes':[
              "ref_index"

          ],
          "db_alias":"Websites_classification"
          }

class URLBow_fulltxt(Document):
    short_genres=ListField(StringField(required=True))
    bow= DictField(field=IntField(),required=True)
    bow_index=IntField(required=True)

    meta={'collection':'URLBow_fulltxt',
          'indexes':[
              'short_genres',
              "bow_index"

          ],
          "db_alias":"Websites_classification"
          }

class URLAllGram(Document):
    short_genres=ListField(StringField(required=True))
    attr_map= DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'URLAllGram',
          'indexes':[
              'short_genres',
              "ref_index"

          ],
          "db_alias":"Websites_classification"
          }
