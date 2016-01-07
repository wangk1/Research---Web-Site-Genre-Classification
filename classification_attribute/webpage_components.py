__author__ = 'Kevin'
from db.db_model.mongo_websites_models import URLToGenre
from .word_based import BagOfWords,TextUtil

from bs4 import BeautifulSoup

def extract_title(reference_db,db_cls):
    """
    Extract title from some webpage in URLToGenre and save it to the db_cls database

    reference db's object must have url and ref_index attributes
    :param db_cls:
    :return:
    """

    bow_transformer=BagOfWords()
    for ref_object in reference_db.no_cache():
        url=ref_object.url
        ref_index=ref_object.ref_index

        page=URLToGenre.objects.get(url=url).only("page").page

        page_soup=BeautifulSoup(page,"html.parser")

        title=page_soup.title.string

        #bag of word it
        title_bow=bow_transformer(title)

        title_bow=TextUtil.remove_stop_words(TextUtil.stem(title_bow))

        #store into db
        db_cls(url=url,ref_index=ref_index,attr_map=title_bow).save()

def extract_meta_data():
    """
    Extract meta data descriptions(name=description) and keywords

    :return:
    """

    #soup.find_all("meta",{"name":"description"})[0]["content"]
    #soup.find_all("meta",{"name":"Description"})[0]["content"]

    #soup.find_all("meta",{"name":"description"})[0]["content"]
    #soup.find_all("meta",{"name":"Description"})[0]["content"]