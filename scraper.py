__author__ = 'Kevin'

from scraper.web_scraper import WebScraper,WebPageInfo,GenreScraper
from db.db_model.mongo_websites_models import URLToGenreAlexa300K,URLQueue,URLToGenre
from db import DBQueue


def scrape_alexa_300k():
    queue=DBQueue(queue_cls=None,queue_name="alexa_300k")

    scraper=WebScraper(queue=queue)

    with open("C:\\Users\\wangk1\\Desktop\\Research\\research\\top_300k_alexa\\top_300k.txt") as top_300k_file:
        scraper.scrape_pipeline((WebPageInfo(url=i[:-1]) for i in top_300k_file),URLToGenreAlexa300K)

def scrape_genre_data():
    queue=DBQueue(queue_cls=URLQueue,queue_name="genre_data")

    scraper=GenreScraper(queue=queue)

    start=queue.get_location()
    scraper.scrape_pipeline((WebPageInfo(url=URLQueue.objects.get(number=i).document.url) for i in range(start,URLQueue.objects.count())),URLToGenre,
                            start=start)

if __name__=="__main__":
    scrape_genre_data()