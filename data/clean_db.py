
"""
Module Responsible for hosting files that clean up databases.

"""

def remove_references_in_attr_map(db_cls):
    """
    This method is Useless, it does not work. DB must be
    cleaned with native JS code.
    :param db_cls:
    :return:
    """
    return

    for db_obj in db_cls.objects.no_cache():
        attr_map=db_obj.attr_map

        if "_cls" in attr_map:
            del attr_map["_cls"]

        if "_ref" in attr_map:
            del attr_map["_ref"]

        db_cls.objects(url=db_obj.url).update(attr_map=attr_map)