__author__ = 'Kevin'

import settings
import datetime

with open(settings.LOG_ERROR_PATH,mode='a') as ehandle, open(settings.LOG_BAD_URL,mode='a') as blog:
    day='\nNew Log on date: '+str(datetime.date.today())+'---------\n'
    ehandle.write(day)
    blog.write(day)

class Logger:

    @staticmethod
    def error(msg,file=settings.LOG_ERROR_PATH):
        msg='ERROR: {}'.format(msg)
        try:
            print(msg)
        except:
            print("can't print for some reason")
        with open(file,mode='a') as ehandle:
            ehandle.write(msg+'\n')

    @staticmethod
    def info(msg):
        msg='Info: {}'.format(msg)
        try:
            print(msg)
        except:
            print("Can't print for some reason")

    @staticmethod
    def debug(msg):
        msg='Debug: {}'.format(msg)
        try:
            print(msg)
        except:
            print("Can't print for some reason")

    @staticmethod
    def debug(msg):
        if settings.DEBUG:
            print(msg)
