__author__ = 'Kevin'

import urllib3
from bs4 import BeautifulSoup
import time,random
from util.base_util import *
from util.Logger import Logger
__all__=['Request']

headers={'user-agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}

class Request:
    #self.http
    http=urllib3.PoolManager(headers=headers)
    bad_count=0

    """Must wait so as to not trigger
    """
    def _randomized_wait(self):
        time.sleep(random.random()*settings.WAIT_TIME)

    def get(self, url):
        self._randomized_wait()
        response=None
        try:
            response= self.http.request('GET',url,timeout=settings.TIME_OUT)
            self.bad_count=0
        except:
            self.bad_count+=1

            # wait and sleep until we get an answer
            if self.bad_count >= settings.REQUEST_EXCEPTION_UNTIL_TEST_CONNECTION:
                while(not self.testInternet()):
                    Logger.info('Waiting for internet')
                    time.sleep(2)

                response= self.http.request('GET',url,timeout=settings.TIME_OUT)
                self.bad_count=0

        return response

    def testInternet(self):
        try:
            self.http.request('GET',settings.INTERNET_TEST_ADDR)

            return True
        except:
            return False