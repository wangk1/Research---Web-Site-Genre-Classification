__author__ = 'Kevin'
from classification.classification_res import RightResultsIter,WrongResultsIter

if __name__ == "__main__":
    res_path="classification_res\\supervised\\supervised_summary_chi_sq_50000"
    classifiers=["LogisticRegression"]


    print(len([i for i in RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier="")]))