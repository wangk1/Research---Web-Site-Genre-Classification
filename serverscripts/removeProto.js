
//Javascript for extracting URL from mongodb

db = db.getSiblingDB('Websites_mutual');

var cursor=db.MutualInformation.find();


cursor.forEach(function(doc) {
    var bow=doc.bow;

    if ("__proto__" in bow) {
        delete doc.bow["__proto__"]
        print("Deleted from "+doc.short_genre);

        db.MutualInformation.update({short_genre:doc.short_genre},{$set:{bow:bow}});

    }



});