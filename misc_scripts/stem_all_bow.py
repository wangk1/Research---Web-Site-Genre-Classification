from collections import Counter
import itertools
import db.db_collections.mongo_collections as coll
from classification_attribute.word_based import EnglishStemmer
from util import base_util as util

__author__ = 'Kevin'


def stem_bow():
    url_queue_coll=coll.URLQueue()
    url_bow_coll=coll.URLBow()
    porter_stemmer=EnglishStemmer()

    for index in range(101,52160):

        queue_obj=url_queue_coll.select(number=index).find_one()

        url_obj=queue_obj.document

        util.print_safe("On index {}, {}".format(index,url_obj.url))

        stemmed_dict=Counter()
        if hasattr(url_obj,"original") and url_obj.original:


            bow_obj=url_bow_coll.select(url=url_obj.url).find_one()

            if not bow_obj:
                continue

            print("Updating")

            stemmed_dict.update(itertools.chain(
                    #stem the keys and map it to a list only have the key x times. since Counter .update only
                    #takes in list of repeating sequence
                    *map(lambda item_tuple:[porter_stemmer(item_tuple[0])]*item_tuple[1],
                         #filter out proto and ref
                         ((k,v) for k,v in bow_obj.bow.items() if k != "__proto__" and k != "_ref")
                        )
            ))

            bow_obj.update(bow=stemmed_dict)
            bow_obj.save()
            bow_obj.reload()