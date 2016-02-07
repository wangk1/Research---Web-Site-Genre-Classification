__author__ = 'Kevin'

from db.db_model.mongo_websites_models import URLToGenre
from util.base_util import unreplace_dot_url

def grab_urls_and_genres():
    file_name="url_list.txt"

    line_template="{}:::{}\n"
    with open(file_name,encoding="latin-1",mode="w") as url_file_handle:


        for url_obj in URLToGenre.objects(original=True):
            genres_list=[g.genre for g in url_obj.genre]

            url_file_handle.write(line_template.format(unreplace_dot_url(url_obj.url),":::".join(genres_list)))




