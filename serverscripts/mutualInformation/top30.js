
/*Get top 30 word from each genre along with their respective joint count




*/

db=db.getSiblingDB('Websites_mutual');

//sort by obj's 1st property and obj2's second property
function sortByValueDecrement(obj,obj2) {
    return obj2[1]-obj1[1];

}

//Extract top 30 word for each short_genre
db.WordGenreJoin.find().forEach(function(doc) {
    db.Top30WordGenre.insert();

    var decSortedCount=

});