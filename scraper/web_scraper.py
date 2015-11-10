import collections
import itertools

from .base_scraper import BaseScraper
from service.RequestService import Request
from db import DBQueue
from db.db_collection_operations import Genres,GenreMetaData
from db.db_model.mongo_websites_models import EmbeddedGenre, URLToGenreAlexa300K
from .alexa_scraper import AlexaScraper
from .dmoz_scraper import DMOZScraper
from util.base_util import normalize_genre_string,unreplace_dot_url, replace_dot_url


__author__ = 'Kevin'


WebPageInfo=collections.namedtuple("WebPageInfo",("URL",""))

class WebScraper(BaseScraper):

    def __init__(self,queue):
        super().__init__(Request())

        assert isinstance(queue,DBQueue)
        #queue for iteration over samples
        self.queue=queue

        self.alexa_scraper=AlexaScraper()
        self.dmoz_scraper=DMOZScraper()

    def scrape_pipeline(self,webpageinfo_iterable,output_collection):
        """

        """

        for webpageinfo_obj in itertools.islice(webpageinfo_iterable,self.queue.get_location()):
            assert isinstance(webpageinfo_obj,WebPageInfo)
            url=unreplace_dot_url(WebPageInfo.URL)

            #first get the webpage
            page=self.get_page(url)

            dot_replaced_url=replace_dot_url(url)

            #then get the urls's genres from alexa and dmoz that are EXACT matches and convert from string -> genre coll objects
            alexa_genre_refs=Genres.create_genres(self.alexa_scraper.query_url(url),dot_replaced_url)
            dmoz_genre_refs=Genres.create_genres(self.dmoz_scraper.query_url(url),dot_replaced_url)

            #convert from genres -> embedded genres for more info and storage in genre_metadata
            alexa_embedded_ref_list=(EmbeddedGenre(type="url",genre=g_ref,result_type="alexa") for g_ref in alexa_genre_refs)
            dmoz_embedded_ref_list=(EmbeddedGenre(type="url",genre=g_ref,result_type="dmoz") for g_ref in dmoz_genre_refs)

            #Create the genre metadata
            genre_metadata=GenreMetaData.create_genremetadata(itertools.chain(alexa_embedded_ref_list,dmoz_embedded_ref_list),dot_replaced_url)

            #finally put page in collection
            output_collection(genres_data=genre_metadata,url=dot_replaced_url,original=True,page=page).save()

            #update reference so we don't go over the same again
            self.queue.increment_location()



class WebScraperMapper:
    """
    A mapper

    """
    pass