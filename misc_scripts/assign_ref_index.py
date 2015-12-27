__author__ = 'Kevin'

from db.db_model.mongo_websites_models import URLQueue,URLToGenre

def assign_ref_index_to_each_url():
    """
    This script assigns ref index to each url in URLToGenre. Each one that does not have a ref index is also assigned
        one after

    :return:
    """
    max_ref_index=0
    print("Giving each url object in urltogenre ref indexes")
    for count,url_bow_obj in enumerate(URLQueue.objects):
        count %500==0 and print("Done {} in updating existing pages in URLQueue".format(count))

        url=url_bow_obj.document.url

        URLToGenre.objects.get(url=url).update(ref_index=url_bow_obj.number)

        if url_bow_obj.number>max_ref_index:
            max_ref_index=url_bow_obj.number

    for count,url_obj in enumerate(URLToGenre.objects(ref_index=-2)):
        count %500==0 and print("Done {} in updating existing pages not in URLQueue".format(count))

        max_ref_index+=1
        URLToGenre.objects.get(url=url_obj.url).update(ref_index=max_ref_index)

    print("Done")