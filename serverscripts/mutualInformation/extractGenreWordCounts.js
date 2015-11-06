/*Counts the number of times a certain feature, word, occurs with a genre, the class.

Note that the GenreWordJoinCount collection will be emptied
*/

//The databases
db_web_classification = db.getSiblingDB('Websites_classification');
db_web_mutual= db.getSiblingDB('Websites_mutual');

var cursor=db_web_classification.TrainSetBow.find();

print("Emptying WordCount_training and GenreCound_training for new counts");
print(db_web_mutual.GenreCount_training.remove({}));
print(db_web_mutual.WordCount_training.remove({}));

var c=0;
while (cursor.hasNext()) {
    var trainingSetObj=cursor.next();
    if (c%100==0)
        print(c);

    bow=trainingSetObj.bow;
    genre=trainingSetObj.short_genre;

    bow_new={count:1};
    for (w in bow) {
        if (w==="_ref" || w==="__proto__" || w===null ||!w) {
            continue;
        }

        db_web_mutual.WordCount_training.update({word:w},{$inc:{count:bow[w]}},{upsert:true});
        bow_new["bow."+w]=bow[w]

    }

    db_web_mutual.GenreCount_training.update({genre:genre},{$inc:bow_new},{upsert:true});

    ++c;
}