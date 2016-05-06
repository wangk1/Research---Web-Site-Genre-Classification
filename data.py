__author__ = 'Kevin'
from data.db_to_pickle import db_to_pickle
from db.db_model.mongo_websites_classification import *
from db.db_model.mongo_websites_models import URLBow
from data.text_to_bow import *


if __name__=="__main__":
    extract
    #db_to_pickle(WebpageMetaBOW,"metadata")
    #db_to_pickle(WebpageTitleBOW,"title")
    #URLToGenre_to_bow()
    db_to_pickle(URLBow_fulltxt,"fulltxt")
