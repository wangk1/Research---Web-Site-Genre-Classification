__author__ = 'Kevin'

import settings
from util import base_util
from db.db_model.mongo_websites_models import *
from db.database import MongoDB
from scraper.web_scrape_oldr import WebScraper
from scraper.dmoz_scraper import DMOZScraper

class PipeLine:

    @classmethod
    def get_top_from_boxer(cls):

        return cls

    @classmethod
    def save_top_urls_to_mongo(cls):
        with open(settings.OUTPUT_FILE,encoding='ISO-8859-1') as input:

            for webpage in input:
                #dictionary of webpage properties
                webpage_dict=cls.__get_url_properties_and_sanitize(webpage)
                MongoDB.save_modify_url(**webpage_dict)

        return cls

    @classmethod
    def create_url_queue(cls):
        for num,URL_document in enumerate(URLToGenre.objects):
                MongoDB.push_to_queue(num,URL_document)

        return cls

    @classmethod
    def scrape_urls(cls):

        position=MongoDB.get(MetaData,'position',type='queue')

        WebScraper().scrape_links(position)

        return cls

    @staticmethod
    def __get_url_properties_and_sanitize(boxer_line):
        split_elements=boxer_line.split(settings.BOXER_DELIMITER)

        url=base_util.replace_dot_url(split_elements[settings.INDEX_OF_URL])

        genre=split_elements[settings.INDEX_OF_GENRE].replace('_','/',1)

        desc=split_elements[settings.INDEX_OF_DESC]

        return {'url':url,'genre':[{'genre':genre,'alexa':{}}],'desc':desc}

    @classmethod
    def random_pick_dmoz(cls):
        return DMOZScraper().scrape()

    @classmethod
    def start(cls):
        MongoDB.connect(settings.HOST_NAME,settings.PORT)

        return cls
