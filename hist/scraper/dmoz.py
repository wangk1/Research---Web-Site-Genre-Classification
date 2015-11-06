__author__ = 'Kevin'

"""
DMOZ Scraper

"""

import functools
import random

from bs4 import BeautifulSoup

import service.RequestService as RequestService
from .scraper import Scraper
from db.database import MongoDB
from util.exception_annotations import fail_safe_web
from util.base_util import *
from util.Logger import Logger



#get dmoz url
dmoz_home='http://www.dmoz.org/';
DMOZ_SEARCH_QUERY='http://www.dmoz.org/search?q={}'

__all__=['DMOZ']

class DMOZ(Scraper):
    #self.dmoz
    #self.exclusions
    #self.mongo

    def __init__(self):
        self.http=RequestService.Request()
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

    """
    Search DMOZ for website, scrap the genre.
    """
    def search(self,url):
        dmoz_url_query=DMOZ_SEARCH_QUERY.format(url)

        dmoz_query_page=self.http.get(dmoz_url_query).data.decode('utf-8')

        query_results=(li for ol in BeautifulSoup(dmoz_query_page,'html.parser').find_all('ol',{'class':'site'})
                            for li in ol.find_all('li'))

        #The individual links
        for position,query_result_li in enumerate(query_results):
            #url on dmoz
            query_url=query_result_li.find_all('a')[0]['href']

            #category
            #dmoz webiste has a div in the li, there is an anchor tag in div that is the category url
            query_category=query_result_li.find_all('div',{'class':'ref'})[0].find_all('a')[0]['href']

            #if exact match
            if hasattr(query_result_li,'class') and query_result_li['class'] == 'star':
                pass

            #not exact match, we can elect to record it too
            else:
                if position<settings.QUERY_LIMIT:
                    pass

