__author__ = 'Kevin'

"""
DMOZ Scraper

"""

import functools
import random

from bs4 import BeautifulSoup

import service.RequestService as RequestService
from scraper.base_scraper import BaseScraper
from db.database.mongodb_connector import MongoDB
from util.exception_annotations import fail_safe_web,fail_safe_encode
from util.Logger import Logger
from db.db_model.mongo_websites_models import *
from util.base_util import *
import db.db_collections.mongo_collections as coll

#get dmoz url
dmoz_home='http://www.dmoz.org/';
DMOZ_SEARCH_QUERY_CATEGORY='http://www.dmoz.org/search?q=u%3A%22{}%22&start=0&type=more&all=no&cat=all'

__all__=['DMOZScraper']

class DMOZScraper(BaseScraper):
    #self.dmoz
    #self.exclusions
    dmoz_search_query_url="http://www.dmoz.org/search?q=u%3A%22{}%22&start=0&all=no&cat=all"

    def __init__(self):
        super().__init__(RequestService.Request(wait_time=0))
        self.exclusions=set()

    def scrape(self):
        home=self.http.get(dmoz_home)

        home_page_links=self._scrapeHomeAndGetLinks(home.data)

        #visit each link in homepage and dig down
        #for url in home_page_links:
        i=0
        while i<settings.NUM_RANDOM_WEBPAGE:
            result=self._scrapPage(home_page_links[random.randint(0,len(home_page_links)-1)])

            if result is not None and MongoDB.get_url_object(result['url']) is None:
                i+=1
                try:
                    page=utf_8_safe_decode(self.http.get(result['url']).data)

                    MongoDB.save_modify_url(page=page,**result)

                    Logger.info("Completed: "+result['url'])
                except Exception as ex:
                    Logger.error(ex)

    @fail_safe_web
    def _scrapeHomeAndGetLinks(self,home):
        soup_home=BeautifulSoup(home,'html.parser')
        soup_home.encode_contents(encoding='utf-8')

        #get the 1/3 column slices containing the major url links we want
        selectionColumns=[s for s in map(lambda t: t.encode('utf-8'),soup_home.body.find_all('div',{'class': 'one-third'}))]

        #grab the major category of link from each 1/3 column slice
        def _getHomeHyperLinks(col):
            greaterCat=BeautifulSoup(col,'html.parser').find_all('span')

            catAnchor=[]
            for cat in greaterCat:
                catAnchor.extend([anchor['href'] for anchor in cat.find_all('a')])

            return catAnchor

        list_of_urls=(_getHomeHyperLinks(col) for col in selectionColumns)

        #flatten each list of columns of url paths
        url_path=functools.reduce(lambda acc,list:acc+list,list_of_urls,[])

        return url_path

    @fail_safe_web
    def _scrapPage(self,url):
        page_html=self.http.get(dmoz_home+url[1:]).data

        soup_page=BeautifulSoup(page_html,'html.parser')

        leaf_webpages=self._findLeaf(soup_page.find_all('ul',{'class','directory-url'}),url)

        #find all list object in ul and combine together
        more_categories=[li for ul in soup_page.find_all('ul',{'class':'directory dir-col'})
                            for li in ul.find_all('li')]

        rand=random.randint(0,len(more_categories))

        if(rand==len(more_categories)):
            return leaf_webpages[random.randint(0,len(leaf_webpages)-1)]

        else:
            return self._scrapPage(more_categories[random.randint(0,len(more_categories)-1)].a['href'])


    def _findLeaf(self,external_links_ul,genre):
        if(len(external_links_ul)==0):
            return []

        #list of tuple of webpage url and its description
        webpage_info=[(li.a['href'], li.text) for ul in external_links_ul
                                        for li in ul.find_all('li')]

        leaf_webpages=[{'url':page[0],'desc':re.sub('\r|\n|\t','',page[1].strip()),'genre':[{'genre':genre,'dmoz':{}}]} for page in webpage_info]


        return leaf_webpages

    def query_url(self,url):
        """
        Query DMOZ for the exact genre matches for the url.

        Returns normalized genre, aka any leading / or trailing / is removed

        :param: url: The url to be queried
        :return: a list of genre strings found

        """
        url=unreplace_dot_url(url)

        query_page=self.get_page(self.dmoz_search_query_url.format(url))

        if query_page is None or query_page.strip()==0:
            raise DoesNotExist("Failed to fetch query page")

        if query_page is None:
            return []

        ols=BeautifulSoup(query_page,'html.parser').find_all('ol',{'class':'site'})

        if len(ols)==0:
            return []

        query_results=(li for ol in ols
                            for li in ol.find_all('li'))

        categories=[]
        #test to see which matches
        for pos,query_li in enumerate(query_results):
            if pos < settings.QUERY_LIMIT_RESULT_SITES:

                div=query_li.find_all('div',{'class':'ref'})[0]

                # if the url we queried for matches the dmoz search
                if take_away_protocol(url) == take_away_protocol(div.text[3:].split('\xa0')[0]):
                    categories.append(normalize_genre_string(div.a['href']))

        return categories


    def __has_genre(self,url_obj,genre):
        if not hasattr(url_obj,'genre_data') or len(GenreMetaData.objects(url=url_obj['url'])) is 0 \
                and len(url_obj['genre_data']['genres'])is 0:
            self.__create_genre_metadata(url_obj)


        #find the genre of the url_obj whose string genre value without first and last forward slash is the same as the
        #genre we are testing for. If there is none, return None
        return next((genre_dict['genre']['genre']  \
                   for genre_dict in GenreMetaData.objects(url=url_obj['url'])[0]['genres'] \
                        if normalize_genre_string(genre_dict['genre']['genre']) == normalize_genre_string(genre) and genre_dict['type']=='dmoz'),None)


    def __create_genre_metadata(self,url_obj):

        genres=[]
        for genre in url_obj['genre']:
            genres.append(EmbeddedGenre(type='alexa',count=1,genre=genre))

        genre_meta_obj=GenreMetaData(url=url_obj['url'],genres=genres)
        genre_meta_obj.save()

        URLToGenre.objects(url=url_obj['url']).update_one(genre_data=genre_meta_obj)
        url_obj.reload()

        return True


