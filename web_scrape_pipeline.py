__author__ = 'Kevin'

import multiprocessing as mp
import queue

import db.db_collections.mongo_collections as coll
import settings
from util import base_util
from db.db_model.mongo_websites_models import *
from db.database.mongodb_connector import MongoDB
from scraper.web_scraper import WebScraper
from scraper.dmoz_scraper import DMOZScraper
from util.Logger import Logger
from db import DBQueue_old


class PipeLine:

    @classmethod
    def get_top_from_boxer(cls):

        return cls

    @classmethod
    def save_top_urls_to_mongo(cls,webpages):

        for webpage in webpages:
            #dictionary of webpage properties
            webpage_dict=cls.__get_url_properties_and_sanitize(webpage)
            MongoDB.save_modify_url(**webpage_dict)

        return cls

    @classmethod
    def create_url_queue(cls):
        sorted_coll=URLToGenre.objects.order_by('_id')
        counter=391417#96461
        total=391417

        try:
            while counter<total:
                MongoDB.push_to_queue(counter,sorted_coll[counter])
                if counter %10000==0:
                    print('{} done'.format(counter))
                counter+=1
        except:
            pass

        print('stopped at {}'.format(counter))
        print('last url: {}'.format(sorted_coll[counter]['url']))

        return cls

    @classmethod
    def scrape_urls(cls):

        position=MongoDB.get(MetaData,'position',type='queue')

        WebScraper().scrape_links(position)

        return cls

    @classmethod
    def scrape_urls_multiproc(cls):
        #current position
        pos=MongoDB.get(MetaData,'position',type='queue')
        #current cap
        cap=pos

        process_queue=queue.Queue(maxsize=settings.NUM_PROCESSES)

        #creates all the necessary processes
        for p_num in range(0,settings.NUM_PROCESSES):
            p=mp.Process(target=WebScraper().scrape_links_from_position,args=[cap])
            #get curresponding objects
            process_queue.put(p)

            cap+=settings.NUM_URLS_PER_PROCESS

            #now start
            p.start()

        head=process_queue.get()
        #wait and create new processes as needed
        while(pos<MongoDB.count(URLQueue)):
            head.join()

            if not head.exitcode ==0:
                Logger.error('Error with Process, terminating')
                return

            #update counter
            MongoDB.increment_url_counter(settings.NUM_URLS_PER_PROCESS)

            p=mp.Process(target=WebScraper().scrape_links_from_position,args=[cap])
            process_queue.put(p)
            p.start()

            #increase both cap and current position
            cap+=settings.NUM_URLS_PER_PROCESS
            pos+=settings.NUM_URLS_PER_PROCESS
            head=process_queue.get()


        print(p.exitcode)

        return cls



    @classmethod
    def __create_process_and_start(cls,**kwargs):
        p=mp.Process(**kwargs)
        p.start()

        return p


    @staticmethod
    def __get_url_properties_and_sanitize(boxer_line):
        split_elements=boxer_line.split(settings.BOXER_DELIMITER)

        url=base_util.replace_dot_url(split_elements[settings.INDEX_OF_URL])

        genre=split_elements[settings.INDEX_OF_GENRE].replace('_','/',1)

        desc=split_elements[settings.INDEX_OF_DESC]

        return {'url':url,'genre':[{'genre':genre,'alexa':{}}],'desc':desc}

    @classmethod
    def random_pick_dmoz(cls):
        return DMOZScraper().scrape()

    @classmethod
    def __create_genre_metadata(cls,url_obj):

        genres=[]
        for genre in url_obj['genre']:
            genres.append({'type':'dmoz','count':1,'genre':genre})

        genre_meta_obj=GenreMetaData(url=url_obj['url'],genres=genres)
        genre_meta_obj.save()

        URLToGenre.objects(url=url_obj['url']).update_one(genre_data=genre_meta_obj)

        return True

    @classmethod
    def start(cls):
        MongoDB.connect(settings.HOST_NAME,settings.PORT)

        return cls
