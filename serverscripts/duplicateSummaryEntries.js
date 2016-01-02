//Get object ids of all the duplicates in the URLBow collection

db = db.getSiblingDB('Websites');

rec=db.URLBow.aggregate([
  { $group: {
    _id: { ref_index: "$ref_index" },   // replace `name` here twice
    uniqueIds: { $addToSet: "$_id" },
    count: { $sum: 1 }
  } },
  { $match: {
    count: { $gte: 2 }
  } },
  { $sort : { count : -1} }
]);

print(JSON.stringify(rec,null,2));