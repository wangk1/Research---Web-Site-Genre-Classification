__author__ = 'Kevin'



def _mixed_effect(X,vocab_vect,feature_selector):
    feature_selector.fit_transform(X)
    selected_indices=[feature_selector.get_support()]

def mixed_effect_chisq():
    pass