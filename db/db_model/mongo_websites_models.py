__author__ = 'Kevin'

from mongoengine import *
from db.db_collections.mongo_collection_base import MongoCollection


class Genre(DynamicDocument):
    genre=StringField(required=True,unique=True)
    dmoz=DictField(default={})
    urls=ListField(StringField())

    meta={'collection':'Genre',
          'indexes':[
              'genre'

          ]}

class GenreTest(DynamicDocument):
    genre=StringField(required=True,unique=True)
    dmoz=DictField(default={})
    urls=ListField(StringField())

    meta={'collection':'GenreTest',
          'indexes':[
              'genre'

          ]}

class ShortGenre(DynamicDocument):
    short_genre=StringField(required=True,unique=True)
    genres=ListField(ReferenceField(Genre,reverse_delete_rule=PULL),default=[],required=True)

    meta={'collection':'ShortGenre',
          'indexes':[
              'short_genre'

          ]}

class EmbeddedGenre(EmbeddedDocument):
    type=StringField(required=True)
    count=IntField(required=True,default=-1)
    genre=ReferenceField(Genre,required=True)
    result_type=StringField(default='NA')

class EmbeddedGenreTest(EmbeddedDocument):
    type=StringField(required=True)
    #count=IntField(required=True,default=-1)
    genre=ReferenceField(GenreTest,required=True)
    result_type=StringField(default='NA')

class GenreMetaData(DynamicDocument):
    genres=ListField(EmbeddedDocumentField(EmbeddedGenre),default=[])
    url=StringField()

    meta={'collection':'GenreMetaData',
          'indexes':[
              'url'

          ]
          }

class GenreMetaDataTest(DynamicDocument):
    genres=ListField(EmbeddedDocumentField(EmbeddedGenreTest),required=True)
    url=StringField()

    meta={'collection':'GenreMetaDataTest',
          'indexes':[
              'url'

          ]
          }

class URLToGenre(DynamicDocument):
    genre=ListField(ReferenceField(Genre))
    url=StringField(unique=True,required=True)
    genres=ReferenceField(GenreMetaData)
    original=BooleanField(required=True,default=False)
    ref_index=IntField(required=False,default=-2)

    #the following are optional fields that may exist
    parent=ListField()
    #page=StringField()

    meta={'collection':'URLToGenre',
          'indexes':[
              'url',
              'ref_index'

          ]}

class URLToGenreAlexa300K(DynamicDocument):
    url=StringField(unique=True,required=True)
    genres_data=ReferenceField(GenreMetaData,required=True)
    original=BooleanField(required=True,default=False)
    page=StringField()
    ranking=IntField()

    #the following are optional fields that may exist
    parent=ListField()

    meta={'collection':'URLToGenre300K',
          'indexes':[
              'url'

          ]}

class URLQueue(Document):
    number=IntField(required=True,unique=True)
    document=ReferenceField(URLToGenre,required=True,reverse_delete_rule=CASCADE)

    meta={'collection':'URLQueue',
          'indexes':[
              'number'

          ]}

class MetaData(Document):
    position=IntField(required=True,default=0)

    type=StringField(required=True,default='queue')

    meta={'collection':'MetaData'
          }

class DMOZQueryNotFound(Document):
    url=StringField(required=True)

    meta={'collection':'DMOZNotFound',
          'indexes':[
              'url'

          ]
          }


class Analytics(DynamicDocument):
    name=StringField(required=True,unique=True)

    #There can be many other document key depending on the analytic type

    meta={'collection':'Analytics',
          'indexes':[
              'name'

          ]
          }

class QueryResults(DynamicDocument):
    url=StringField(required=True,unique=True)
    site=StringField(required=True)

    #There can be many other document key depending on the analytic type

    meta={'collection':'QueryResults',
          'indexes':[
              'url'

          ]
          }

class CategoricalBOW(Document):
    genre=StringField(required=True,unique=True)
    bow=MapField(field=IntField(), default={})

    meta={'collection':'CategoricalBOW',
          'indexes':[
              'genre'

          ]
          }

class URLBow(Document):
    url=StringField(required=True,unique=True)
    bow=MapField(field=IntField(), default={})
    short_genres=ListField(StringField(),default=[])
    index=IntField()

    meta={'collection':'URLBow',
          'indexes':[
              'url'

          ]
          }

class MutualInformationGenres(Document):
    genre=StringField(required=True)
    bow=MapField(field=FloatField(),default={})

    meta={'collection':'MutualInformationGenres',
          'indexes':[
              'genre'

          ],
          "db_alias":"Websites_mutual"
          }

class Top30WordGenre(Document):
    short_genre=StringField(required=True)
    bow=MapField(field=FloatField(),default={})

    meta={'collection':'Top30WordGenre',
          'indexes':[
              'short_genre'

          ],
          "db_alias":"Websites_mutual"
          }

class WordGenreJoint(Document):
    short_genre=StringField(required=True)
    bow=MapField(field=FloatField(),default={})

    meta={'collection':'WordGenreJoint',
          'indexes':[
              'short_genre'

          ],
          "db_alias":"Websites_mutual"
          }

class TrainSetBow(Document):
    short_genre=StringField(required=True)
    attr_map=MapField(field=IntField(),default={})
    ref_index=IntField(required=True)

    meta={'collection':'TrainSetBow',
          'indexes':[
              'ref_index'

          ],
          "db_alias":"Websites_classification"
          }

class TestSetBow(Document):
    short_genre=StringField(required=True)
    attr_map=MapField(field=IntField(),default={})
    ref_index=IntField(required=True)

    meta={'collection':'TestSetBow',
          'indexes':[
              'ref_index'

          ],
          "db_alias":"Websites_classification"
          }

class TrainTestWordCount(Document):
    type=StringField()
    vocab=ListField(StringField())

    meta={'collection':'TrainTestWordCount',
          'indexes':[
              'type'

          ],
          "db_alias":"Websites_classification"
          }

class GenreCount_training(Document):
    genre=StringField(required=True)
    count=IntField(required=True,default=0)
    bow=DictField(field=IntField(default=0),required=True,default={})

    meta={'collection':'GenreCount_training',
          'indexes':[
              'genre'

          ],
          "db_alias":"Websites_mutual"
          }

class WordCount_training(Document):
    word=StringField(required=True)
    count=IntField(required=True,default=0)

    meta={'collection':'WordCount_training',
          'indexes':[
              'word'

          ],
          "db_alias":"Websites_mutual"
          }

class TopWordGenre(Document):
    genre=StringField(required=True)
    bow=DictField(field=IntField(default=0))

    meta={'collection':'TopWordGenre',
          'indexes':[
              'genre'

          ],
          "db_alias":"Websites_mutual"
          }