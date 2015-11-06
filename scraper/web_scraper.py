__author__ = 'Kevin'
import random
import re

from bs4 import BeautifulSoup

from .scraper import Scraper
from service.RequestService import Request
from db.database.mongodb_connector import MongoDB
from db.db_model.mongo_websites_models import *
import settings
from . import config
from util.exception_annotations import fail_safe_web
from util import base_util
from util.Logger import Logger


class WebScraper(Scraper):
    #self.http


    def __init__(self):
        self.http=Request()

    @fail_safe_web
    def scrape(self,url,parent):
        Logger.debug('Starting url scrap for {}'.format(url))
        config.last_url_and_parent=url+', {}'.format('' if parent==None else parent)

        new_url=base_util.unreplace_dot_url(url)

        response=self.http.get(new_url)
        Logger.debug('Got URL')
        if not hasattr(response,'data') and new_url.startswith('www.'):
            new_url=new_url.replace('www.','http://')

            response=self.http.get(new_url)

            if not hasattr(response,'data'):
                new_url=new_url.replace('http://','http://www.')
                response=self.http.get(new_url)


        if hasattr(response,'data'):
            body=base_util.utf_8_safe_decode(response.data)

        else:
            Logger.error('No data associated with '+new_url)
            raise AttributeError(new_url+':::No data')

        return body,new_url

    @fail_safe_web
    def scrape_link_and_child(self,parent_url):
        parent_url=base_util.replace_dot_url(parent_url)
        webpage_body,parent_url=self.scrape(base_util.unreplace_dot_url(parent_url),None)

        #exit if failed to scrap website
        if webpage_body is None:
            return

        Logger.debug('Saving Parent')
        MongoDB.save_page(url=parent_url,page=webpage_body)
        Logger.info('Completed page: '+parent_url)

        #Now, we grab the childs of this webpage
        all_ahref=[base_util.combine_parent_rel_link(parent_url,a.attrs['href']) for a in BeautifulSoup(webpage_body,'html.parser', from_encoding="utf-8").find_all('a') if 'href' in a.attrs]

        child_urls=random.sample(all_ahref,settings.GET_X_CHILD) if len(all_ahref)>=settings.GET_X_CHILD else all_ahref

        #get rid of bad normalization
        if not re.match('^www[.].*$',parent_url):
            Logger.info('Updating bad url for {}'.format(parent_url))
            MongoDB.update_url(base_util.normalize_url(parent_url),parent_url)

        if len(child_urls) > 0:

            #get the childs, child urls is a subset of all urls
            for child_url in child_urls:
                Logger.debug('Get Child {}'.format(child_url))
                child_page=self.scrape(child_url,parent_url)

                if child_page is None:
                    exploredset=set()
                    tries=0
                    for url in set(all_ahref)^(exploredset):
                        if tries==settings.MAX_RETRIES:
                            Logger.info('Max retrie number exceeded')
                            break

                        Logger.info("trying new url: "+url)

                        child_page=self.scrape(url,parent_url)

                        if child_page is not None:
                            break
                        exploredset.add(url)

                        tries+=1

                if child_page is not None:
                    Logger.debug('Saving Child {}'.format(child_url))
                    MongoDB.save_modify_url(url=base_util.replace_dot_url(child_url),parent=[MongoDB.get_url_object(parent_url)],genre=[],page=child_page)
                    Logger.info('Completed page: '+child_url)

    def scrape_links(self,pos):

        doc_object=MongoDB.get(URLQueue,'document',number=pos)

        while doc_object is not None:
            self.scrape_link_and_child(doc_object['url'])
            pos=MongoDB.increment_url_counter()

            doc_object=MongoDB.get(URLQueue,'document',number=pos)
    '''
    Use multiple process to scrap webpage

    '''
    def scrape_links_from_position(self,pos):
        MongoDB.connect(settings.HOST_NAME,settings.PORT)
        links=self.__get_next_urls(pos)


        Logger.info(links)
        for link in links:
            self.scrape_link_and_child(link)

        Logger.debug('Process job completed')
        return 0

    '''
    Get the next x urls from url queue depending on the current position number
    '''
    def __get_next_urls(self,curr_pos):
        doc_objs=MongoDB.get_m(URLQueue,'document',number__in=list(range(curr_pos,curr_pos+settings.NUM_URLS_PER_PROCESS)))

        return [doc_obj['url'] for doc_obj in doc_objs]