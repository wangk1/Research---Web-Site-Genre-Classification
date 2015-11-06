__author__ = 'Kevin'
from .Logger import Logger
import settings
import scraper.config
from util import base_util
import traceback

class ExceptionHandler:
    #self.count
    def __init__(self):
        self.count=0

    def handle(self,ex):
        raise NotImplementedError('Must implement own exception handler.')

    def to_string(self):
        raise NotImplementedError('Must implement own exception handler.')

class MongoHandler(ExceptionHandler):
    #self.type

    def __init__(self):
        ExceptionHandler.__init__(self)
        self.type='Mongo Issue'

    def handle(self,ex,*arg,**kwargs):
        self.count+=1


        try:
            err_msg=scraper.config.last_url_and_parent+':::Reason(Mongo): {}\n'.format(base_util.get_class(ex))

            Logger.error(err_msg)

            with open(settings.LOG_BAD_URL,mode='a') as blog:
                blog.write(err_msg)
                blog.write(scraper.config.last_url_and_parent+'\t\t\t {}\n'.format(traceback.format_exc()))
        except:
            Logger.error('Bad URL')

class WebHandler(ExceptionHandler):
    #self.type

    def __init__(self):
        ExceptionHandler.__init__(self)
        self.type='WebScraping Issue'

    def handle(self,ex,*arg,**kwargs):
        self.count+=1

        Logger.error('WebScraping Issue: '+str(ex))

        try:
            with open(settings.LOG_BAD_URL,mode='a') as blog:
                blog.write(scraper.config.last_url_and_parent+':::Reason: {}\n'.format(str(ex)))
        except:
            Logger.error('Failed to write URL to bad url file.')

class DefaulExceptionHandler(ExceptionHandler):
    def __init__(self):
        ExceptionHandler.__init__(self)
        self.type='Default'

    def handle(self,ex,*arg,**kwargs):
        self.count+=1
        Logger.error(str(ex))