__author__ = 'Kevin'

from web_scrape_pipeline import PipeLine
from db.db_collection_operations.url_queue_ops import *
from collections import Counter
import os,itertools
from classification_attribute.word_based import EnglishStemmer
from db.db_model.mongo_websites_models import TrainTestWordCount,TrainSetBow
from db.db_model.mongo_websites_models import URLBow

"""
models=MongoDB(settings.HOST_NAME,settings.PORT)
genres=models.save_modify_url('aurl',*[{'genre':'test1','alexa':12,'dmoz':10000},{'genre':'test2'},{'genre':'test3'}])
genres=models.save_modify_url('gaurl',*[{'genre':'test1','alexa':12,'dmoz':10000},{'genre':'test5','alexa':13,'dmoz':130000},{'genre':'test2'},{'genre':'test3'}])
"""

if __name__ =='__main__':
    """
    url_coll=coll.URLToGenre()

    counter=Counter()
    for c,url_bow_obj in enumerate(coll.URLBow()):
        if c%1000==0:
            print(str(c))

        url_obj=url_coll.select(url=url_bow_obj["url"]).find_one()

        for genre in url_obj.genre:
            counter.update([util.normalize_genre_string(genre.genre,level=1)])

    print(counter)
    """


    # c=set()
    # for count,train_bow_obj in enumerate(TrainSetBow.objects):
    #     count%1000==0 and print(count)
    #     c|=set(k for k in train_bow_obj.bow.keys() if len(k)<10)
    #
    # TrainTestWordCount(type="train",vocab=c).save()
    #
    # pass
    print("hellos")



'''
    not_found=set()
    with open("C:\\Users\Kevin\\Google Drive\\Webpage Classification Research\\NOT_FOUND.txt",encoding="ISO-8859-1") as not_found_txt:
        for url in not_found:
            not_found.add(url)

    all_webpage={}
    with open(settings.OUTPUT_FILE,encoding="ISO-8859-1") as all_url:

        for url in all_url:
            if url in not_found:
                url_split=url.split(':::')
                webpage_dict={}
                webpage_dict['url']=url_split[0]
                webpage_dict['genre']=[url_split[1]]
                webpage_dict['desc']=url_split[2]

'''



