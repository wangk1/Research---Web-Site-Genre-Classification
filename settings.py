__author__ = 'Kevin'

DEBUG=True

"""
Http request settings
"""
#multiplies this by random number b/w 0 and one generated by random
WAIT_TIME=6

TIME_OUT=3

REQUEST_EXCEPTION_UNTIL_TEST_CONNECTION=3
#test in case the network is out
INTERNET_TEST_ADDR='www.google.com'

#only retries each website once if first time failed
RETRIE_UPON_ERROR=1

NO_DATA_RETRIE=1

MIN_PAGE_SIZE=500 #min page size in char, gets rid of much 404 errors

"""
Database settings

"""
USE_DB_TYPE="MONGO"

USE_DB='Websites'
HOST_NAME='localhost'
PORT=27017

SAVE_LOCATION_PER=10    #save the location in boxer text per x urls scraped


URL_GROUP_SIZE=20 #Will batch process urls

"""
Boxer_update file read settings
"""
GET_TOP=250

BOXER_LOCATION='C:\\Users\\Kevin\\Google Drive\\Webpage Classification Research\\Boxer_update.txt'

OUTPUT_FILE='C:\\Users\\Kevin\\Google Drive\\Webpage Classification Research\\TOP_'+str(GET_TOP)+'_OF_EACH_CATEGORY'

NOT_FOUND='C:\\Users\\Kevin\\Google Drive\\Webpage Classification Research\\NOT_FOUND.txt'

BOXER_DELIMITER=':::'

INDEX_OF_URL=0
INDEX_OF_GENRE=1
INDEX_OF_DESC=2

URL_DOT_REPLACE='&dot;'

"""
DMOZ query setting

"""

QUERY_LIMIT_RESULT_SITES=20  #for queries, how many websites should we look at until we give up

QUERY_LIMIT_RESULT_CAT=5  #for queries, how many websites should we look at until we give up
"""
Child scraping

"""

GET_X_CHILD=4
MAX_RETRIES=3


"""
Logging

"""
LOG_IMPORTANT_PATH="important.txt"

LOG_ERROR_PATH='error.txt'

LOG_INFO_PATH='info.txt'

LOG_BAD_URL='bad_url.txt'

"""
Random Queries

"""

NUM_RANDOM_WEBPAGE=10000

"""
Multiprocessing
"""
NUM_PROCESSES=6

NUM_URLS_PER_PROCESS=1

TIME_OUT=10

"""
The url queue

"""


"""
Query settings
"""
TOP_X_CATEGORIES=6


"""
Pickling

"""
PICKLE_DIR="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir"