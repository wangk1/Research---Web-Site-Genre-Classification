__author__ = 'Kevin'

from db.db_model.mongo_websites_models import URLQueue,URLToGenre
from db.db_model.mongo_websites_classification import URLAllGram
from db.db_model.mongo_queue_models import Queue_full_page
from data.util import pickle_obj

def assign_ref_index_to_each_url():
    """
    This script assigns ref index to each url in URLToGenre. Each one that does not have a ref index is also assigned
        one after

    :return:
    """
    max_ref_index=0
    print("Giving each url object in urltogenre ref indexes")
    for count,url_bow_obj in enumerate(URLQueue.objects):
        #count %500==0 and print("Done {} in updating existing pages in URLQueue".format(count))

        url_obj=url_bow_obj.document

        url_obj.update(ref_index=url_bow_obj.number)

        if url_bow_obj.number>max_ref_index:
            max_ref_index=url_bow_obj.number

    for count,url_obj in enumerate(URLToGenre.objects(ref_index=-2)):
        count %500==0 and print("Done {} in updating existing pages not in URLQueue".format(count))

        max_ref_index+=1
        URLToGenre.objects.get(url=url_obj.url).update(ref_index=max_ref_index)

    print("Done")

def reassign_ref_index_to_url_ngram():

    url_to_ref_index={}
    for c,url_ngram_obj in enumerate(URLAllGram.objects):
        c%1000==0 and print("Done with {}".format(c))

        full_page_obj=Queue_full_page.objects(number=url_ngram_obj.ref_index)[0]

        url_to_genre_obj=URLToGenre.objects(url=full_page_obj.url)[0]

        if full_page_obj.url in url_to_ref_index:
            raise AttributeError("URL already exists in mapping")

        url_to_ref_index[full_page_obj.url]=url_to_genre_obj.ref_index

    #pickle the object just in case


    for url,ref_index in url_to_ref_index.items():
        full_page_obj=Queue_full_page.objects(url=url)[0]

        #update full page object
        full_page_obj.update(ref_index=ref_index)
        #update urlallgram object
        URLAllGram.objects(full_page_obj.number)[0].update(ref_index=ref_index)

