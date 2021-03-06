__author__ = 'Kevin'


"""
Http request settings
"""
#multiplies this by random number b/w 0 and one generated by random
WAIT_TIME_MULTIPLIER=0

TIME_OUT=4

REQUEST_EXCEPTION_UNTIL_TEST_CONNECTION=3
#test in case the network is out
INTERNET_TEST_ADDR='www.google.com'

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

BOXER_DELIMITER=':::'

INDEX_OF_URL=0
INDEX_OF_GENRE=1
INDEX_OF_DESC=2

URL_DOT_REPLACE='&dot;'

"""
DMOZ query setting

"""

QUERY_LIMIT=5

"""
Child scraping

"""

GET_X_CHILD=4
MAX_RETRIES=10


"""
Logging

"""
LOG_ERROR_PATH='error.txt'

LOG_INFO_PATH='info.txt'

LOG_BAD_URL='bad_url.txt'

"""
Random Queries

"""

NUM_RANDOM_WEBPAGE=10000