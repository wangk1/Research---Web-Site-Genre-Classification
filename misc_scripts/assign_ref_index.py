__author__ = 'Kevin'

from db.db_model.mongo_websites_models import URLQueue,URLToGenre,URLBow
from db.db_model.mongo_websites_classification import URLAllGram
from db.db_model.mongo_queue_models import Queue_full_page
from data.util import pickle_obj

def global_ref_id():

    """
    Had an issue with reference ids not matching b/w url allgrams, urltogenre pages, and summary in url bow.

    This resolves the issur and uses urlbow's index for all ids

    :return:
    """
    urls=set()
    largest_ref_index=0
    for c,url_summary_obj in enumerate(URLBow.objects):
        c%1000==0 and print("Done with {}".format(c))
        url=url_summary_obj.url
        urls.add(url)
        ref_index=url_summary_obj.ref_index

        if ref_index is None:
            ref_index=largest_ref_index+1
            url_summary_obj.update(ref_index=ref_index)

        if ref_index > largest_ref_index:
            largest_ref_index=ref_index

        URLToGenre.objects(url=url).update(ref_index=ref_index)

    print("Done with normal ones, just finishing off the rest of URLTOGenre")
    for url_to_genre_obj in URLToGenre.objects:
        if url_to_genre_obj.url in urls:
            continue

        largest_ref_index+=1
        url_to_genre_obj.update(ref_index=largest_ref_index)
