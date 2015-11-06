__author__ = 'Kevin'

from scraper.alexa_scraper import AlexaScraper
from scraper.dmoz_scraper import DMOZScraper

if __name__=="__main__":
    scraper=DMOZScraper()

    print(scraper.query_url("www.google.com"))