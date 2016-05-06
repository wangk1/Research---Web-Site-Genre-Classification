__author__ = 'Kevin'
from mongoengine import *

#current db's alias
db_aliase="Websites_classification"

class TrainSet_urlFullTextBow(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TrainSet_urlFullTextBow',
          'indexes':[
              "ref_index"

          ],
          "db_alias":db_aliase
          }

class TestSet_urlFullTextBow(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TestSet_urlFullTextBow',
          'indexes':[
              "ref_index"

          ],
          "db_alias":db_aliase
          }

class TrainSet_urlAllGram(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TrainSet_urlAllGram',
          'indexes':[
              "ref_index"

          ],
          "db_alias":db_aliase
          }

class TestSet_urlAllGram(Document):
    short_genre=StringField(required=True)
    attr_map=DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)

    meta={'collection':'TestSet_urlAllGram',
          'indexes':[
              "ref_index"

          ],
          "db_alias":db_aliase
          }

class URLBow_fulltxt(Document):
    short_genres=ListField(StringField(required=True))
    attr_map= DictField(field=IntField(),required=True)
    ref_index=IntField(required=True)
    url=StringField(required=True)

    meta={'collection':'URLBow_fulltxt',
          'indexes':[
              "ref_index",
              "url"
          ],
          "db_alias":db_aliase
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
          "db_alias":db_aliase
          }

class WebpageTitleBOW(Document):
    short_genres=ListField(StringField(required=True))
    attr_map= DictField(field=IntField(),default={})
    ref_index=IntField(required=True)

    meta={'collection':'WebpageTitleBOW',
          'indexes':[
              'short_genres',
              "ref_index"


          ],
          "db_alias":db_aliase
          }

class WebpageMetaBOW(Document):
    short_genres=ListField(StringField(required=True))
    attr_map= DictField(field=IntField(),default={})
    ref_index=IntField(required=True)

    meta={'collection':'WebpageMetaBOW',
          'indexes':[
              'short_genres',
              "ref_index"


          ],
          "db_alias":db_aliase
          }