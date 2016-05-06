
__author__="Kevin"

from db.db_model.mongo_websites_models import URLToGenre,URLBow
from db.db_model.mongo_websites_classification import URLBow_fulltxt
from classification_attribute import word_based
from util.base_util import normalize_genre_string

"""
This module has code that converts Full webpage text to bag of words
"""

def URLToGenre_to_bow():
    bow=word_based.BagOfWords()

    all_ref_index=set(i.ref_index for i in URLBow.objects.no_cache())

    URLBow_fulltxt.objects.delete()
    for c,db_obj in enumerate(URLToGenre.objects.no_cache()):
        if c%1000==0:
            print("Done with {}".format(c))

        if not hasattr(db_obj,"original") or not db_obj.original \
            or not hasattr(db_obj,"page") or not db_obj.page:
            continue

        ref_id=db_obj.ref_index
        if ref_id not in all_ref_index:
            continue

        page=db_obj.page

        if isinstance(page,list):
            print("{} is a list updating".format(ref_id))
            page="".join(page)
            #db_obj.update(page=page)

        try:
            word_dict=bow.get_word_count(page)
        except:
            print("Skipped {}".format(ref_id))
            continue
        short_genres=[normalize_genre_string(genre.genre) for genre in db_obj.genre]

        URLBow_fulltxt(ref_index=ref_id,attr_map=word_dict,short_genres=short_genres,\
                       url=db_obj.url).save()
