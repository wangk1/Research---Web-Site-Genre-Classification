__author__ = 'Kevin'
import settings
import util.base_util as util
import re

def get_dictionary_from_top_250_line(boxer_line):
    split_elements=boxer_line.split(settings.BOXER_DELIMITER)

    url=util.replace_dot_url(re.sub('^[0-9]+ ','',split_elements[settings.INDEX_OF_URL])).strip()
    url=url[0].lower()+url[1:]

    genre=split_elements[settings.INDEX_OF_GENRE].replace('_','/',1)

    desc=split_elements[settings.INDEX_OF_DESC]

    return {'url':url,'genre':[{'genre':genre,'alexa':{}}],'desc':desc}