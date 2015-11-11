__author__ = 'Kevin'

from scraper.base_scraper import BaseScraper,safe_scrape
from service.RequestService import Request
from util.base_util import unreplace_dot_url

from bs4 import BeautifulSoup

class AlexaScraper(BaseScraper):
    ignored_top_level={"World"}

    alexa_template="http://www.alexa.com/siteinfo/{}"

    def __init__(self):
        super().__init__(Request())

    @safe_scrape
    def query_url(self,url):
        """
        Query the url in alexa, will automatically unreplace the &dot;

        Returns normalized genre, aka any leading / or trailing / is removed

        :raise AssertionError: Assertion error(None page or empty page)
        :param url: url to be
        :return genrestring: genre string
        """
        url=unreplace_dot_url(url)

        page=self.get_page(AlexaScraper.alexa_template.format(url))

        all_genre_strings=[]
        if page is None or page.strip() == "":
            raise AssertionError("The page is either empty or none")
        else:
            page_soup=BeautifulSoup(page,"html.parser")
            #ignore world
            link_table=page_soup.find(id="category_link_table").find("tbody")

            all_genre_tr=link_table.find_all("tr")

            for tr in all_genre_tr:
                span=tr.find("span")

                all_genre_strings.append('/'.join([genre_component_link.string for genre_component_link in span.find_all("a")]))

        return all_genre_strings
