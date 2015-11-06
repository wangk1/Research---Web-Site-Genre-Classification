__author__ = 'Kevin'
import settings
import re,codecs

def boxer_generator():
    with codecs.open(settings.BOXER_LOCATION,'r',encoding='ISO-8859-1') as boxer:
                #,open(settings.OUTPUT_FILE,'w',encoding='ISO-8859-1') as output:
            prev_cat=''
            start=0

            for linenum,line in enumerate(boxer):

                #extract the category via regex, look for ::: before and ::: after string
                category=re.search('(?<={0}).*(?={0})'.format(settings.BOXER_DELIMITER),line)

                #swap category, we start the tracker from the beginning
                if category and category.group(0) != prev_cat:
                    start=linenum
                    prev_cat=category.group(0)

                #only output if we are at the or above nth popular website in the category where n is the popularity ranking
                #we want to stop at
                if linenum-start+1 <= settings.GET_TOP:
                    yield str(linenum-start+1)+' '+line


