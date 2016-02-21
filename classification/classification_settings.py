import functools

from sklearn.feature_selection import SelectKBest,chi2,f_classif
from collections import namedtuple
from data import LearningSettings

"""
Module containing settings and variables for Supervised classification


"""

#genres to filter out with filter genre methods
ignore_genre={
    "World",
    "id",
    "desc",
    "page",
    "parent",
    "url",
    "genres",
    "Kids_and_Teens",
    "Kids",
    "Regional",
    "Home",
    "News"
}

"""
Global Settings for classification
"""
global_settings=namedtuple("GlobalSettings",
                               ("train_set_size","res_dir","pickle_dir","print_res",
                                "k_folds","feature_selector_name","k_fold_random_state")
                               ) (
        train_set_size=50000,
        res_dir="classification_res", #store results into this directory
        pickle_dir="pickle_dir", #where to pickle objects such as matrices to be used later
        print_res=True, #print res into file in res_dir directory
        k_folds=10,
        feature_selector_name="anova",
        k_fold_random_state=744149
        #print_if_less_than_x_weight=4 #not used yet

    )


"""
CLASSIFICATION SETTINGS
"""
setting_summary=LearningSettings(type="supervised",dim_reduction=global_settings.feature_selector_name,num_attributes=0,feature_selection="summary",
                         pickle_dir=global_settings.pickle_dir,res_dir=global_settings.res_dir)

setting_url=LearningSettings(type="supervised",dim_reduction=global_settings.feature_selector_name,num_attributes=0,feature_selection="url",
                         pickle_dir=global_settings.pickle_dir,res_dir=global_settings.res_dir)

setting_meta=LearningSettings(type="supervised",dim_reduction=global_settings.feature_selector_name,num_attributes=0,feature_selection="metadata",
                         pickle_dir=global_settings.pickle_dir,res_dir=global_settings.res_dir)

setting_title=LearningSettings(type="supervised",dim_reduction=global_settings.feature_selector_name,num_attributes=0,feature_selection="title",
                         pickle_dir=global_settings.pickle_dir,res_dir=global_settings.res_dir)

#select which data sets to use simultaneously
settings=[setting_summary, #100000
              setting_url, #60000
              setting_meta, #60000
              setting_title #30000
              ]

"""
Individual settings
"""
for setting in settings:
        setting.result_file_label="no_reg_kid_hom_new"
        setting.threshold=4
        setting.ll_ranking=False #DO NOT USE NOT SUPPORTED
        setting.num_attributes={
                                10000#,20000,30000,40000,50000,60000,70000,80000,100000,120000,130000,160000,200000
                                }
settings[0].num_attributes={100000}
settings[1].num_attributes={60000}
settings[2].num_attributes={60000}
settings[3].num_attributes={30000}

"""
Weights
"""
weights=namedtuple("weights",("num_classifiers","weights_range","stepping","fixed",
                              "fixed_weight")) (
    num_classifiers=len(settings),
    weights_range=(0,1),
    stepping=0.2,
    fixed=True, #Set true to lock wieghts to fixed_weight tuple instead of using range
    fixed_weight=(0.6,0.2,0.4,0.4) #Tuple of (w1,w2...) for each classifier

)

"""
Feature Selection stradegy
"""
feature_selection_strategy={
    "chi_sq":functools.partial(SelectKBest,chi2),
    "anova":functools.partial(SelectKBest,f_classif)

}