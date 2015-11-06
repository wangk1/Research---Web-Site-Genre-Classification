__author__ = 'Kevin'

from service.RequestService import Request
from util.base_util import *


def safe_scrape(func):
    """
    Catch any attribute errors that may be associated with querying an alexa or dmoz page not present
    and send back an empty list instead

    :param func:
    :return:
    """
    def wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except (AttributeError,AssertionError) as e:
            return []

    return wrapper

class BaseScraper:

    def __init__(self,request):
        assert isinstance(request,Request)
        self.http=request

    def get_page(self,url):
        """
        Get the utf-8 decoded html page of url.
        :return:
        """
        return utf_8_safe_decode(self.http.get_data(url))