__author__ = 'Kevin'

import settings
import datetime
from warnings import warn

# with open(settings.LOG_ERROR_PATH,mode='a') as ehandle, open(settings.LOG_BAD_URL,mode='a') as blog:
#     day='\nNew Log on date: '+str(datetime.date.today())+'---------\n'
#     ehandle.write(day)
#     blog.write(day)

DEBUG=False
class Logger:

    def __init__(self,module=""):
        self.module_name=module


    def error(self,msg,file=settings.LOG_ERROR_PATH):
        msg='ERROR: {}'.format(msg)
        try:
            warn(msg)
        except:
            print("can't print for some reason")
        with open(file,mode='a') as ehandle:
            ehandle.write(msg+'\n')

    def warn(self,msg):
        msg='WARNING: {}'.format(msg)
        try:
            print(msg)
        except:
            print("Can't print for some reason")

    def info(self,msg):
        msg='Info: {}'.format(msg)
        try:
            print(msg)
        except:
            print("Can't print for some reason")

    def debug(self,msg):
        if not DEBUG:
            return
        msg='Debug: {}'.format(msg)
        try:
            print(msg)
        except:
            print("Can't print for some reason")

