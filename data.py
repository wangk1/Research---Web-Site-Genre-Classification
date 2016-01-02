__author__ = 'Kevin'
from data.db_to_pickle import db_to_pickle
from db.db_model.mongo_websites_classification import URLAllGram
from db.db_model.mongo_websites_models import URLBow



if __name__=="__main__":
    db_to_pickle(URLAllGram,"url")
    db_to_pickle(URLBow,"summary")
