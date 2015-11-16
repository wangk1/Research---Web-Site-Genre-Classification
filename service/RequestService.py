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

    header_combo={"https://","http://"}

    def __init__(self,wait_time=settings.WAIT_TIME,timeout=settings.TIME_OUT):
        #used for timeouts
        self.bad_count=0
        self.timeout=timeout
        self.http=urllib3.PoolManager(headers=Request.headers,timeout=urllib3.Timeout(self.timeout),retries=settings.RETRIE_UPON_ERROR)
        #this is a number that is multiplied by a random number for a number of miliseconds to wait.
        #essentially, this extends/shrinks the wait time, allowing user to adjust the wait time
        self.wait_time=wait_time


    def _randomized_wait(self):
        """
        Don't want to trigger a lockout on some webpages, so we randomize the wait

        :return:
        """
        time.sleep(random.randint(self.wait_time,self.wait_time+10))

    @fail_safe_web
    def get_data(self,url):

        #retry twice if the url doesn't work
        for retrie in range(0,settings.NO_DATA_RETRIE):

            for header in self.header_combo:
                try:
                    response=self.get((header if not url.startswith(header) else "")+url)

                except Exception as ex:
                    pass

                if hasattr(response,"data") and len(response.data)>settings.MIN_PAGE_SIZE:
                    break
                else:
                    response=None

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