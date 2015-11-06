__author__ = 'Kevin'
from db.db_collections.mongo_collections import *
import operator

def get_top_X_word_genre(x=30):
    """
    Grab top X word of each short_genre in joint between Genre and Word and store it in Top30WordGenre collection

    30 is default
    :return:
    """
    top_30=Top30WordGenre()
    mi=MutualInformation()

    for mi_obj in mi.iterable():
        #try:
        print(mi_obj.short_genre)

        top_30_bow=dict(
            sorted(
                filter(lambda x: not operator.itemgetter(0)(x).isdigit(),mi_obj.bow.items())
            ,key=operator.itemgetter(1),reverse=True)[:x]
        )

        top_30.create(short_genre=mi_obj.short_genre,bow=top_30_bow)
        #except Exception as ex:
            #print("Failed to create top 30 for genre {}, reason: {}".format(mi_obj.short_genre,str(ex)))
