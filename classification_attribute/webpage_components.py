__author__ = 'Kevin'
from db.db_model.mongo_websites_models import URLToGenre
from .word_based import BagOfWords,TextUtil
from util.Logger import Logger
from data.util import genre_normalizer

from bs4 import BeautifulSoup

comp_logger=Logger(__name__)

def extract_title(reference_db_cls,db_cls):
    """
    Extract title from some webpage in URLToGenre and save it to the db_cls database

    reference db's object must have url and ref_index attributes
    :param db_cls:
    :return:
    """
    comp_logger.info("Extracting from the database {}, putting into {}".format(reference_db_cls,db_cls))

    bow_transformer=BagOfWords()
    title_not_exists=0
    for c,ref_object in enumerate(reference_db_cls.objects.no_cache()):
        c%10==0 and comp_logger.info("Done with {} titles".format(c))

        url=ref_object.url
        ref_index=ref_object.ref_index
        short_genres=genre_normalizer(ref_object.short_genres,dim=1)

        page=URLToGenre.objects(url=url).only("page")[0].page

        page_soup=BeautifulSoup(page,"html.parser")

        try:
            title=page_soup.title.string

            #bag of word
            #title_bow=bow_transformer.get_word_count(title) if title and title.strip() else {}

        except (AttributeError,ValueError):
            title_not_exists+=1
            title_bow={}

        #store into db
        #db_cls(ref_index=ref_index,attr_map=title_bow,short_genres=short_genres).save()

    comp_logger.info("The title does not exists in {} instances".format(title_not_exists))

def extract_meta_data(reference_db_cls,db_cls):
    """
    For selected webpages in URLToGenre:

    Extract meta data descriptions(name=description) and keywords and form bag of words representation with it.

    Store it into a database

    :return: None
    """
    comp_logger.info("Extracting from the database {}, putting into {}".format(reference_db_cls,db_cls))

    bow_transformer=BagOfWords()
    not_found_data=0
    for c,ref_object in enumerate(reference_db_cls.objects.no_cache()):
        c%10000==0 and comp_logger.info("Done with {} MetaDatas".format(c))

        url=ref_object.url
        ref_index=ref_object.ref_index
        short_genres=genre_normalizer(ref_object.short_genres,dim=1)

        page=URLToGenre.objects(url=url).only("page")[0].page

        page_soup=BeautifulSoup(page,"html.parser")

        contents=[]
        try:
            for meta_data_desc in page_soup.find_all("meta",{"name":"description"}):
                contents.append(meta_data_desc["content"])

            for meta_data_desc in page_soup.find_all("meta",{"name":"Description"}):
                contents.append(meta_data_desc["content"])

            for meta_data_desc in page_soup.find_all("meta",{"name":"keywords"}):
                contents.append(meta_data_desc["content"])

            contents=" ".join(contents if contents else "")
            #meta_bow=bow_transformer.get_word_count(contents) if contents and contents.strip() else {}

            if not len(contents):
                not_found_data+=1
        except (KeyError,AttributeError,ValueError):
            not_found_data+=1
            meta_bow={}

        #store into db
        #db_cls(ref_index=ref_index,attr_map=meta_bow,short_genres=short_genres).save()

    comp_logger.info("The MetaData does not exists in {} instances".format(not_found_data))