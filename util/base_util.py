__author__ = 'Kevin'
import settings
import re
'''''
Utility module providing basic utility functionalities

'''''

"""
Normalize URL into a standard www.something.com format

Removes http:// or https://

DON"T USE

"""
def normalize_url(url):
    url=re.sub('http[s]?://','',url.lower())
    url='www.'+url if url.find('www.')==-1 and url.find('www&dot;')==-1 else url
    url=replace_dot_url(url)

    return url

def take_away_protocol(url):

    url=re.sub('^http[s]?://','',url)
    url=re.sub('^www.','',url)

    if url.strip()=='':
        return url

    return url[:-1] if url[-1]=='/' else url

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

def remove_query(url):
    m=url.split('?')[0]

    return m

def utf_8_safe_decode(txt,errors='ignore'):
    """
    UTF 8 decode with ignore the input txt. Note that this method will throw an Attribute error
        if the txt is not bytes.

    """

    return txt.decode('utf-8',errors) if not txt is None else None

def make_sure_utf_8(txt,errors='ignore'):
    return txt.encode('utf-8',errors=errors).decode('utf-8',errors=errors)

def print_safe(txt):
    try:
        print(txt)
    except Exception as e:
        print(e)

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

"""
Get rid of forward slash preceding and after the genre string.

If none exist, just return the same genre back again

"""
def normalize_genre_string(g,level=-1):

    if g.endswith('/'):
        g=g[:-1]

    if g.startswith('/'):
        g=g[1:]

    # we need to extract until certain level
    if level > -1:
        g='/'.join(g.split('/')[:level])

    return g

"""
Quick method for calculating levenshtein edit distance

"""
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def include_https(url):
    """
    Add https:// infront of the url if it does not have http:// or https://

    :param url:
    :return:
    """

    assert isinstance(url,str)

    if not re.match("http[s]?://",url):
        url="https://"+url

    return url