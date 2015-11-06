__author__ = 'Kevin'
import settings
import re
'''''
Utility module providing basic utility functionalities

'''''

"""
Normalize URL into a standard www.something.com format

Removes http:// or https://

"""
def normalize_url(url):
    url=re.sub('http[s]?://','',url.lower())
    url='www.'+url if url.find('www.')==-1 and url.find('www&dot;')==-1 else url
    url=replace_dot_url(url)

    return url


def combine_parent_rel_link(parent_url,link):
    if link.strip()=='':
        return parent_url

    #for .. or .
    link=link[1:] if re.match('^[.].+',link) else link

    # match relative links
    link='/'+link if re.match('^[^./]+[.](htm[l]?|jsp|php|cfm|rss|asp[x]?)',link)\
        or not (re.match('(^http[s]?://(www[.])?.*)|(^www[.].*$)',link) or link[0]=='/') else link

    #tack to parent if start with /
    return parent_url+link if link[0]=='/' else link

def replace_dot_url(url):
    if url.find('.') != -1:
        return url.replace('.',settings.URL_DOT_REPLACE)

    return url

def utf_8_safe_decode(txt):
    return txt.decode('utf-8','replace')

def unreplace_dot_url(url):
    return url.replace(settings.URL_DOT_REPLACE,'.')

def pop_attr(attr,dictionary):
    return dictionary.pop(attr,None)

def get_class(obj):
    try:
        clz= str(obj.__class__.__name__)
    except:
        clz='Unknown'

    return clz