import collections
import itertools
import summary.summary_util as s_util
from summary.summarizer import Summarizer
from db import DBQueue
from db.db_model.mongo_websites_models import URLToGenre,Summary
import util.base_util as base_util
from classification_attribute.word_based import BagOfWords, TextUtil
from util.Logger import Logger
from service.RequestService import Request

import db.db_collections.mongo_collections as coll
__author__ = 'Kevin'

summarize_logger=Logger(__name__)
def start_bow():
    """
    Bag of word all webpages in URLToGenre database

    Pipeline:
    1. Get genre and page from URLToGenre Object from the mongodb URLQueue
    2. BOW the webpage in URLToGenre Object
    3. Shorten the Genre
    4. Insert the words in bow into the genre in CategoricalBow Mongodb table

    Repeat until exhaustion of URLToGenre Objects

    :return: Nothing!
    """

    queue=DBQueue(None,"summarization")

    #don't trust anything
    summarizer=Summarizer()
    bow=BagOfWords()

    for url_obj in URLToGenre.objects.order_by("ref_index").no_cache():

        try:
            print('New url {}'.format(url_obj.ref_index))

            if not hasattr(url_obj,"original") or not url_obj["original"]:
                continue

            #skip conditionsL it does not have page or it is not an original url
            if not hasattr(url_obj,'page'):
                raise Exception('url {} No has page'.format(url_obj.ref_index))

            #get genre strings
            #register the genre with the short genres for faster retrieval
            genre_string_list=[]
            for g in url_obj.genre:
                normalized_string=base_util.normalize_genre_string(g["genre"])
                genre_string_list.append(normalized_string)

            genre_string_list=list(set(genre_string_list))

            summarize_logger.info("Getting bow rep")
            #get BOW representation
            bow_dict=bow.get_word_count(summarizer.summarize(url_obj.page if isinstance(url_obj.page,str) else base_util.utf_8_safe_decode(url_obj)))

            summarize_logger.info("Update count:"+str(bow_dict))

            if len(bow_dict)==0:
                raise Exception("No count available")

            #store the url bow in urlbow table
            if len(Summary.objects(url=url_obj.ref_index))==0:
                Summary(url=url_obj.url,ref_index=url_obj.ref_index,attr_map=bow_dict,short_genres=genre_string_list).save()
            else:
                print('Exists bow url number {}'.format(url_obj.ref_index))

        except Exception as ex:
            summarize_logger.error(url_obj['url']+":::"+str(ex),"C:/Users/Kevin/Desktop/GitHub/Research/Webscraper/bad_url_summarize_bow.txt")


def collect_bad_url():
    """
    Make bows of websites in the bad url list

    :return:
    """

    queue=DBQueue_old("genre_bow")

    #don't trust anything
    summarizer=Summarizer()
    bow=BagOfWords()
    short_genre_to_genre=coll.ShortGenre()
    url_to_bow=coll.URLBow()
    start_pos=queue.get()

    for c,line in enumerate(open("bad_url_summarize_bow.txt")):
        if c<start_pos:
            continue

        url=line.split(" ")[1].split(":::")[0]

        try:
            print('New url {} num: {}'.format(url,c))

            url_obj=coll.URLToGenre().select(url=url).find_one()

            if not hasattr(url_obj,"original") or not url_obj["original"]:
                print("Not original")
                continue

            #request page anyways, most of the bad pages are due to bad pagess
            data=Request().get_data(base_util.unreplace_dot_url(base_util.unreplace_dot_url(url_obj["url"])))

            if data is None:
                raise Exception('url {} No has page'.format(url))
            else:
                if not hasattr(url_obj,"page") or len(data)>len(url_obj["page"]):
                    print("updating data")
                    data=base_util.utf_8_safe_decode(data)

                    if not hasattr(url_obj,"page"):
                        #save page if the new page is significantly bigger than the old one
                        url_obj.save(page=data)

                    else:
                        url_obj.update(page=data)
                    url_obj.reload()

            if len(data) > len(url_obj.page):
                raise Exception("Inconsistency b/w data and page data")



            #url_obj=repair.genre_to_genre_data(url_obj.document)

            #get genre strings
            #register the genre with the short genres for faster retrieval
            genre_string_list=[]
            for g in url_obj.genre:
                normalized_string=base_util.normalize_genre_string(g["genre"])
                genre_string_list.append(normalized_string)
                short_genre_to_genre.select(short_genre=normalized_string).update(upsert=True,add_to_set__genres=g)

            Logger.info("Getting bow rep")
            #get BOW representation
            bow_dict=bow.get_word_count(summarizer.summarize(url_obj.page if isinstance(url_obj.page,str) else base_util.utf_8_safe_decode(url_obj)))

            if len(bow_dict)<20:
                raise Exception("Words less than 20")

            Logger.info("Update count:"+str(bow_dict))


            #store the url bow in urlbow table
            if not url_to_bow.select(url=url_obj["url"]).find_one():
                url_to_bow.create(url=url_obj["url"],bow=bow_dict,short_genres=genre_string_list)

            else:
                print('Exists bow url number {}'.format(url))

            queue.increment()
        except Exception as ex:
            Logger.error(url_obj['url']+":::"+str(ex),"C:/Users/Kevin/Desktop/GitHub/Research/Webscraper/bad_url_summarize_bow1.txt")



if __name__=="__main__":
    collect_bad_url()


