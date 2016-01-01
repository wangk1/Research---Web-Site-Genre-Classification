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
    for url_summary_obj in URLBow.objects:
        url=url_summary_obj.url
        ref_index=url_summary_obj.ref_index

        URLToGenre.objects(url=url).update(ref_index=ref_index)


