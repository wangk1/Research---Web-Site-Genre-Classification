__author__ = 'Kevin'

from scraper.web_scraper import WebScraper,WebPageInfo
from db.db_model.mongo_websites_models import URLToGenreAlexa300K
from db import DBQueue

if __name__=="__main__":
    queue=DBQueue(queue_cls=None,queue_name="alexa_300k")

    scraper=WebScraper(queue=queue)

    with open("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\top_300k_alexa\\top_300k.txt") as top_300k_file:
        scraper.scrape_pipeline((WebPageInfo(url=i[:-1]) for i in top_300k_file),URLToGenreAlexa300K)