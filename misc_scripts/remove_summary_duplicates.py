__author__ = 'Kevin'
from db.db_model.mongo_websites_models import URLBow,URLToGenre
import json

def remove_summary_duplicates_in_urlbow():

    max_ref_index=391577
    with open("duplicates.json") as duplicates_json_file:
        duplicate_objs_batch=json.load(duplicates_json_file)

        for duplicate_obj_list in duplicate_objs_batch['_firstBatch']:
            ref_index=duplicate_obj_list['_id']['ref_index']

            duplicate_objs=[URLBow.objects.get(id=unique_id) for unique_id in
                                                (unique_ids for unique_ids in
                                                             (id['str'] for id in duplicate_obj_list['uniqueIds'])
                                                 )]

            #update URLToGenre,the first one keeps the index
            for url in map(lambda o:o.url,duplicate_objs[1:]):
                max_ref_index+=1
                URLToGenre.objects.get(url=url).update(ref_index=max_ref_index)
                URLBow.objects.get(url=url).update(ref_index=max_ref_index)

