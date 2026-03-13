def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    num, den = 0, 0
    for sim, r in zip(similarities, ratings):
        if sim > 0:
            num += sim * r 
            den += sim 

    if den > 0:
        return num / den 
    else:
        return 0.0 