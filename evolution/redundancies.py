__author__ = 'Kevin'

import re
import itertools

from util.file_util import *
from db.db_model.mongo_websites_models import *


def save_top_urls_to_mongo():

    with open(settings.OUTPUT_FILE,encoding='ISO-8859-1') as input, open(settings.NOT_FOUND,mode='a',encoding='ISO-8859-1')as not_found:

        skipTo=40000
        for c,webpage in enumerate(itertools.islice(input, skipTo, None),skipTo):
            if c<skipTo:
                input.readline()
                continue


            #dictionary of webpage properties
            webpage_dict=get_dictionary_from_top_250_line(webpage)
            #print(c,webpage_dict['url'])

            if not __find_in_db( webpage_dict['url']):
                bad_url=re.sub('^(h|H)?ttp[s]?://','',webpage_dict['url'])
                bad_url=('www&dot;' if not bad_url.startswith('www&dot;') else '')+bad_url

                if not __find_in_db(bad_url) :
                    if __find_in_db('http://'+bad_url):
                        bad_url='http://'+bad_url

                        _replace_bad_url(bad_url,webpage_dict['url'])
                    elif __find_in_db(re.sub('www&dot;','http://',bad_url)):
                        _replace_bad_url(re.sub('www&dot;','http://',bad_url),webpage_dict['url'])

                    else:
                        try:
                            print('Defintitely Not found {}'.format(webpage_dict['url']))
                        except:
                            print('oops bad printing')
                        not_found.write(webpage_dict['url']+'\n')
                        not_found.flush()
                else:
                    _replace_bad_url(bad_url,webpage_dict['url'])

            if c%1000==0:
                print(str(c)+' done')

def _replace_bad_url(bad_url,url):
    URLToGenre.objects(url=bad_url).update_one(url=url)

    if len(GenreMetaData.objects(url=bad_url))>0:
        GenreMetaData.objects.update(url=url)

def __find_in_db(url):
    return len(URLToGenre.objects(url=url)) > 0



