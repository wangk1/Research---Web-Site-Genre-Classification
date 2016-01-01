
__author__ = 'Kevin'

from classification_attribute.feature_selection import BagOfWords
from classification_attribute.word_based import NGrams
from db.db_model.mongo_queue_models import Queue_full_page
from db.db_model.mongo_websites_models import URLToGenre
from db.db_model.mongo_websites_classification import URLBow_fulltxt,URLAllGram
from db import DBQueue
from util.base_util import normalize_genre_string,unreplace_dot_url
from classification_attribute.url_based import URLTransformer

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


def url_ngram_queue():
    """
    Create ngram database of all the urls in the URLToGenre database that has original flag set to true

    :return:
    """
    #clearing db
    URLAllGram.objects().delete()

    url_model=URLTransformer()

    for c,url_to_genre_obj in enumerate(URLToGenre.objects(original=True)):
        c%1000==0 and print("Done with {}".format(c))

        ref_index=url_to_genre_obj.ref_index
        url=url_to_genre_obj.url

        ngram=url_model.transform(url)

        all_gram_obj=URLAllGram(attr_map=ngram,ref_index=ref_index,short_genres=[normalize_genre_string(genre.genre,1)
                                                                            for genre in url_to_genre_obj.genre]).save()
        for genre in url_to_genre_obj.genre:
            del genre

        if hasattr(url_to_genre_obj,"genre_data") :
            for genre in url_to_genre_obj.genre_data.genres:
                del genre.genre

            del url_to_genre_obj.genre_data

        del url_to_genre_obj
        del all_gram_obj

def field_test():
    for c,obj in enumerate(Queue_full_page.objects):
        c%1000==0 and print("{} done".format(c))


if __name__=="__main__":
    url_ngram_queue()
    #url_ngram_queue()
    #DBQueue(Queue_full_page,"url_ngram_queue").create_queue(URLToGenre.objects(original=True))
    #full_page_bow()