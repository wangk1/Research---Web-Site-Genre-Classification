__author__ = 'Kevin'
import db.db_collections.mongo_collections as coll
from util.file_util import *
import util.base_util as util

'''
Read form top 250 file and create url queue
'''
def read_from_top_250_make_url_queue():
    url_to_genre=coll.URLToGenre()
    output=map(lambda line:url_to_genre.select(url=get_dictionary_from_top_250_line(line)['url']).find_one(),open(settings.OUTPUT_FILE,encoding='ISO-8859-1'))


    coll.URLQueue().create_queue_from_iterable(filter(lambda url_obj: url_obj != None ,output))

def read_from_top_250_mark_original():
    url_to_genre=coll.URLToGenre()

    #filter by those actually in db
    for c,line in enumerate(open(settings.OUTPUT_FILE,encoding='ISO-8859-1')):
        if c%1000==0:
            print("{}".format(c))

        url_obj=url_to_genre.select(url=get_dictionary_from_top_250_line(line)['url']).find_one(only=["url"])

        if url_obj is not None:
            url_obj.modify(original=True)
