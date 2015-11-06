//Javascript for extracting URL from mongodb

db = db.getSiblingDB('Websites');

load('C:/Users/Kevin/Desktop/GitHub/Research/Webscraper/serverscripts/util.js');

var urlToGenreExtractor=function(urlObj){
  var url=urlObj.url;

  //if url obj is valid, print it out
  url && print(unreplacedot(url));
};

var genreMetaDataURLExtractor=function(gMetaObj) {
  gMetaObj['url'] && print(unreplacedot(gMetaObj['url']));

};

db.GenreMetaData.find().forEach(genreMetaDataURLExtractor);