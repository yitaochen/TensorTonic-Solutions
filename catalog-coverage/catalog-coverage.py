def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    unique_items = set()
    for rec in recommendations:
        for item in rec:
            unique_items.add(item)

    return len(unique_items) / n_items