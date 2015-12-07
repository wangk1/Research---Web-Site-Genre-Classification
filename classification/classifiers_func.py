import itertools
import pickle

__author__ = 'Kevin'

from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest,chi2 as chi_sq

from data.util import load_test_matrix,load_train_matrix

genre_count={
'Business': 8495,
'Computers': 6069,
'Arts': 6000,
'Sports': 5858,
'Shopping': 5558,
'Health': 5336,
'Society': 4746,
'Recreation': 4579,
'Science': 3029,
'Games': 2774,
'Home': 2088,
'Adult': 1979,
'Reference': 1921,
'News': 1722,
'Kids': 1633,
'Regional': 1047

}

result_template="classification_res/result_{}.txt"

def logistic(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    """
    Train logistic regression on the training bow and test on test bow

    :param train_bow:
    :param train_labels:
    :param test_bow:
    :return: a, array-like object of prediction
    """
    print("Training logistics")
    logistic_classifier=LogisticRegression()

    logistic_classifier.fit(train_bow,train_labels)

    print("Testing logistics")
    test(logistic_classifier,"logistic",test_bow,test_labels,bow_indexes)

def kNN(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    """
    Train KNN on the training bow and test on test bow

    :param train_bow:
    :param train_labels:
    :param test_bow:
    :return: a, array-like object of prediction
    """
    print("Training kNN")
    knn_classifier=KNeighborsClassifier(n_neighbors=len(genre_count))

    knn_classifier.fit(train_bow,train_labels)

    print("Testing KNN")
    test(knn_classifier,"kNN",test_bow,test_labels,bow_indexes)

def nb(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    print("Training nb")
    mnb_classifier=MultinomialNB()

    mnb_classifier.fit(train_bow,train_labels)

    print("Testing nb")
    test(mnb_classifier,"mNB",test_bow,test_labels,bow_indexes)

def rand_forest(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    print("Training rndForest")
    rf_classifier=RandomForestClassifier()

    rf_classifier.fit(train_bow,train_labels)
    print("Testing rndForest")
    test(rf_classifier,"rf",test_bow,test_labels,bow_indexes)

def decision_tree(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    print("Training decision tree")
    dt_classifier=DecisionTreeClassifier()

    dt_classifier.fit(train_bow,train_labels)
    print("Testing decision tree")
    test(dt_classifier,"dt",test_bow,test_labels,bow_indexes)

def linear_svc(train_bow,train_labels,test_bow,test_labels,bow_indexes):
    print("Training linear svc")
    svc_classifier=LinearSVC()

    svc_classifier.fit(train_bow,train_labels)
    print("Testing linear svc")
    test(svc_classifier,"svc",test_bow,test_labels,bow_indexes)


def test(classifier,classifier_name,test_bow,test_labels,bow_indexes):

    for l in range(0,len(test_labels),500):
        res=classifier.predict(test_bow[l:l+500 if l+500<len(test_labels) else len(test_labels)])
        print_res(classifier_name,
                  labels=test_labels[l:l+500 if l+500<len(test_labels) else len(test_labels)],
                  predictions=res,
                  bow_indexes=bow_indexes[l:l+500])


def load_vocab_vectorizer(train_set,pickle=True,extra_label="default"):
    train_dv=DictVectorizer()

    words=[dict(itertools.chain(*(train_set_obj.attr_map.items() for train_set_obj in train_set.objects())))]
    #fit the dv first
    train_dv.fit(words)
    print("vocab length is {}".format(len(train_dv.feature_names_)))

    del words

    pickle and pickle_dv(train_dv,extra_label)

    return train_dv


def print_res(classifier_name,predictions,labels,bow_indexes):
    with open(result_template.format(classifier_name+"_wrong"),mode="a") as out_wrong, \
        open(result_template.format(classifier_name+"_right"),mode="a") as out_right:
        wrong_labels=(predictions != labels)

        for l in range(0,len(wrong_labels)):
            if wrong_labels[l]==1:
                out_wrong.write("{}, predicted: {}, actual: {}\n".format(bow_indexes[l],predictions[l],labels[l]))
            else:
                out_right.write("{}, predicted: {}, actual: {}\n".format(bow_indexes[l],predictions[l],labels[l]))

PICK_FILE_TEMPLATE="F:/Research Data/pick_vocab_dv_{}"
def pickle_dv(vocab_dv,file_label="default"):
    pickle_file=PICK_FILE_TEMPLATE.format(file_label)
    print("Pickling")
    with open(pickle_file,mode="wb") as pickle_file_handle:
        pickle.dump(vocab_dv,pickle_file_handle)

def unpickle_dv(file_label="default"):
    pickle_file=PICK_FILE_TEMPLATE.format(file_label)
    vocab_dv=None
    with open(pickle_file,mode="rb") as pickle_file_handle:
        vocab_dv=pickle.load(pickle_file_handle)
    return vocab_dv

def classify(*,train_coll_cls,test_coll_cls,k=48000,pickle_label="default"):
    print("Loading vocab")
    vocab_dv=unpickle_dv(pickle_label)#load_vocab_vectorizer()

    print("Loading Samples")
    train_bow,train_labels=load_train_matrix(train_dv=vocab_dv,train_coll_cls=train_coll_cls)
    vectorizer=SelectKBest(chi_sq,k)
    train_bow=vectorizer.fit_transform(train_bow,train_labels)

    test_bow,test_labels,bow_indexes=load_test_matrix(test_dv=vocab_dv,test_coll_cls=test_coll_cls)
    test_bow=vectorizer.transform(test_bow)

    #knn
    kNN(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)
    nb(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)

    decision_tree(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)
    rand_forest(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)
    logistic(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)
    linear_svc(train_bow=train_bow,train_labels=train_labels,test_bow=test_bow,test_labels=test_labels,bow_indexes=bow_indexes)

