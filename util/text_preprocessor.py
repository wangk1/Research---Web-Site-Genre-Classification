__author__ = 'Kevin'

from html2text import html2text
from sumy.parsers import plaintext
from sumy.nlp.tokenizers import Tokenizer
from util.base_util import utf_8_safe_decode


'''
Returns parser object after cleaning html page
    Operations done:
    1. Strip html tags
    2. convert to utf-8

'''
def remove_html(html_page):
    text=html2text(html_page)

    return text

def preprocess(html_page):
    """
    Preprocess the html page, remove the tags and make into utf-8

    :except:
    :param html_page:
    :return: a htmlless text page
    """
    try:
        new_html_page=utf_8_safe_decode(html_page)

    except AttributeError:
        print("Failed to convert page to utf 8, reverting to base page")
        new_html_page=html_page

    return remove_html(new_html_page)


def preprocess_getParser(html_page):
    """
    Preprocess the html page, remove the tags and make into utf-8

    :except:
    :param html_page:
    :return:
    """
    try:
        new_html_page=utf_8_safe_decode(html_page)

    except AttributeError:
        print("Failed to convert page to utf 8, reverting to base page")
        new_html_page=html_page

    pure_txt_page=remove_html(new_html_page)

    return plaintext.PlaintextParser.from_string(pure_txt_page,Tokenizer('english'))

