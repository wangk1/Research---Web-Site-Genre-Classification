__author__ = 'Kevin'


from web_scrape_pipeline import PipeLine
import re
from util.base_util import *
from scraper import web_scrape_oldr

"""
models=MongoDB(settings.HOST_NAME,settings.PORT)
genres=models.save_modify_url('aurl',*[{'genre':'test1','alexa':12,'dmoz':10000},{'genre':'test2'},{'genre':'test3'}])
genres=models.save_modify_url('gaurl',*[{'genre':'test1','alexa':12,'dmoz':10000},{'genre':'test5','alexa':13,'dmoz':130000},{'genre':'test2'},{'genre':'test3'}])
"""
PipeLine.start().scrape_urls().random_pick_dmoz()
