__author__ = 'Kevin'

import settings
import datetime

with open(settings.LOG_ERROR_PATH,mode='a') as ehandle, open(settings.LOG_BAD_URL,mode='a') as blog:
    day='\nNew Log on date: '+str(datetime.date.today())+'---------\n'
    ehandle.write(day)
    blog.write(day)

class Logger:

    @staticmethod
    def error(msg):
        msg='ERROR: {}'.format(msg)

        print(msg)
        with open(settings.LOG_ERROR_PATH,mode='a') as ehandle:
            ehandle.write(msg+'\n')

    @staticmethod
    def info(msg):
        msg='Info: {}'.format(msg)

        print(msg)
