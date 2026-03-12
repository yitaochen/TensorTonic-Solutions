def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    ans = []
    for avg_rating, num_votes in items:
        if num_votes == 0:
            ans.append(global_mean)
        else:
            ans.append((num_votes*avg_rating + min_votes*global_mean)/(min_votes+num_votes))

    return ans 