
__author__ = 'Kevin'

from classification_attribute.feature_selection import BagOfWords
from classification_attribute.word_based import NGrams
from db.db_model.mongo_queue_models import Queue_full_page
from db.db_model.mongo_websites_models import URLToGenre,URLBow
from db.db_model.mongo_websites_classification import *
from db import DBQueue
from util.base_util import normalize_genre_string,unreplace_dot_url
from classification_attribute.url_based import URLTransformer
from classification_attribute.webpage_components import extract_meta_data,extract_title
from util.Logger import Logger
from multiprocessing import Process

attr_logger=Logger(__name__)


def full_page_bow():
    """
    Creates bow of the entire url pages present in URLToGenre that are original pages.

    :return:
    """
    queue=DBQueue(Queue_full_page,"full_page_bow_queue")
    bow_model=BagOfWords()


    for number in range(queue.get_location(),Queue_full_page.objects.count()):
        queue_obj=Queue_full_page.objects.get(number=number)

        url_obj=URLToGenre.objects.get(url=queue_obj.url)

        if number % 1000==0:
            print(number)

        try:

            bow=bow_model.get_word_count(url_obj.page)

            if url_obj.page.strip()=="":
                raise Exception("Bad Page")
        except Exception as ex:
            with open("bad_full_url.txt",mode="a") as out:
                out.write("{}:::{}\n".format(number,str(ex)))
                queue.increment_location()
                continue

        URLBow_fulltxt(bow=bow,bow_index=queue_obj.number,short_genres=[normalize_genre_string(genre.genre,2)
                                                                            for genre in url_obj.genre]).save()
        queue.increment_location()


def create_url_ngram():
    """
    Create ngram database of all the urls in the URLToGenre database that has original flag set to true

    :return:
    """
    #clearing db
    URLAllGram.objects().delete()

    url_model=URLTransformer()

    for c,url_bow_obj in enumerate(URLBow.objects.no_cache()):
        c%1000==0 and print("Done with {}".format(c))

        ref_index=url_bow_obj.ref_index
        url=url_bow_obj.url

        ngram=url_model.transform(url)

        URLAllGram(attr_map=ngram,ref_index=ref_index,short_genres=list(set([normalize_genre_string(genre,1)
                                                                            for genre in url_bow_obj.short_genres]))).save()

def create_meta_DB():
    #clear db
    attr_logger.info("Extracting webpage MetaData")

    WebpageMetaBOW.objects.delete()

    extract_meta_data(URLBow,WebpageMetaBOW)

    attr_logger.info("Done extracting webpage MetaData")

def create_title_DB():
    #clear db
    attr_logger.info("Extracting webpage titles")

    WebpageTitleBOW.objects.delete()

    extract_title(URLBow,WebpageTitleBOW)

    attr_logger.info("Done extracting webpage titles")


if __name__=="__main__":
    p_title=Process(target=create_title_DB)
    p_meta=Process(target=create_meta_DB)

    p_title.start()
    p_meta.start()

    p_meta.join()
    p_title.join()

    #url_ngram_queue()
    #DBQueue(Queue_full_page,"url_ngram_queue").create_queue(URLToGenre.objects(original=True))
    #full_page_bow()