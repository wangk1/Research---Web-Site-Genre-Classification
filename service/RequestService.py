__author__ = 'Kevin'

import urllib3
from bs4 import BeautifulSoup
import time,random
from util.base_util import *
from util.Logger import Logger
from util.exception_annotations import fail_safe_web

__all__=['Request']

user_agent='Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'
connection = 'keep-alive'
AcceptLanguage ='en-US,en;q=0.8'
AcceptEncoding = 'gzip,deflate,sdch'
Accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'


class Request:
    #self.http
    headers=urllib3.make_headers(keep_alive=connection,accept_encoding=AcceptEncoding
                         ,user_agent=user_agent)
    http=urllib3.PoolManager(headers=headers,timeout=urllib3.Timeout(settings.TIME_OUT),retries=settings.RETRIE_UPON_ERROR)
    bad_count=0

    header_combo={"http://","https://"}

    """Must wait so as to not trigger
    """
    def _randomized_wait(self):
        time.sleep(random.random()*settings.WAIT_TIME_MULTIPLIER)

    @fail_safe_web
    def get_data(self,url):
        response=self.get(url)

        if hasattr(response,"data") and response.data.strip() != "":
            return response.data

        for retrie in range(0,settings.NO_DATA_RETRIE):

            for header in self.header_combo:
                if(not hasattr(response,'data') and not url.startswith(header)):
                    response=self.get(header+url)


            if hasattr(response,"data"):
                break

        return response.data

    @fail_safe_web
    def get(self, url):
        self._randomized_wait()
        response=None

        try:

            response= self.http.request('GET',url)
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