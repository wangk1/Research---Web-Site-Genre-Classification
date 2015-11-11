import collections
import itertools

from .base_scraper import BaseScraper
from service.RequestService import Request
from db import DBQueue
from db.db_collection_operations import Genres,GenreMetaData
from db.db_model.mongo_websites_models import EmbeddedGenre, URLToGenreAlexa300K,EmbeddedGenreTest
from .alexa_scraper import AlexaScraper
from .dmoz_scraper import DMOZScraper
from util.base_util import normalize_genre_string,unreplace_dot_url, replace_dot_url
from util.Logger import Logger

webscraper_logger=Logger()


__author__ = 'Kevin'


WebPageInfo=collections.namedtuple("WebPageInfo",("url"))

class WebScraper(BaseScraper):

    def __init__(self,queue):
        super().__init__(Request())

        assert isinstance(queue,DBQueue)
        #queue for iteration over samples
        self.queue=queue

        self.alexa_scraper=AlexaScraper()
        self.dmoz_scraper=DMOZScraper()

    def scrape_pipeline(self,webpageinfo_iterable,output_collection_cls):
        """
        Iterate over WebSiteInfo named tuple iterable. Get the url and grab its genres

        """
        webscraper_logger.debug("Starting webscraper, input from iterable {}, output to {}".format(str(webpageinfo_iterable)
                                                                                                  , output_collection_cls))

        for rank,webpageinfo_obj in itertools.islice(enumerate(webpageinfo_iterable),self.queue.get_location(),None):
            assert isinstance(webpageinfo_obj,WebPageInfo)
            webscraper_logger.debug("Current on rank number {}".format(rank))

            url=unreplace_dot_url(webpageinfo_obj.url)

            #first get the webpage
            page=self.get_page(url)

            if page is None:
                #TODO: catch errors
                webscraper_logger.info("Skippin rank {}".format(rank))
                continue

            webscraper_logger.debug("Found page of length {}".format(len(page)))

            dot_replaced_url=replace_dot_url(url)

            alexa_genre_strings=self.alexa_scraper.query_url(url)
            dmoz_genre_strings=self.dmoz_scraper.query_url(url)

            if len(alexa_genre_strings)+len(dmoz_genre_strings)==0:
                #TODO: catch errors

                webscraper_logger.info("Skippin rank {} due to no genres".format(rank))
                continue

            webscraper_logger.debug("Found {} alexa genres ".format(len(alexa_genre_strings)))
            webscraper_logger.debug("Found {} dmoz genres".format(len(dmoz_genre_strings)))

            #then get the urls's genres from alexa and dmoz that are EXACT matches and convert from string -> genre coll objects
            alexa_genre_refs=Genres.create_genres(alexa_genre_strings,dot_replaced_url)
            dmoz_genre_refs=Genres.create_genres(dmoz_genre_strings,dot_replaced_url)

            #convert from genres -> embedded genres for more info and storage in genre_metadata
            alexa_embedded_ref_list=(EmbeddedGenreTest(type="url",genre=g_ref,result_type="alexa") for g_ref in alexa_genre_refs)
            dmoz_embedded_ref_list=(EmbeddedGenreTest(type="url",genre=g_ref,result_type="dmoz") for g_ref in dmoz_genre_refs)

            #Create the genre metadata
            genre_metadata=GenreMetaData.create_genremetadata([eg for eg in itertools.chain(alexa_embedded_ref_list,dmoz_embedded_ref_list)],dot_replaced_url)


            #finally put page in collection
            output_collection_cls(genres_data=genre_metadata,url=dot_replaced_url,original=True,page=page,ranking=rank).save()
            webscraper_logger.debug("Done, commited to URlToGenreAlexa300k, there are now {} objects"
                                   .format(output_collection_cls.objects.count()))

            #update reference so we don't go over the same again
            self.queue.increment_location()



class WebScraperMapper:
    """
    A mapper

    """
    pass