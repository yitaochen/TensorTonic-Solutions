def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    N, D = len(points), len(points[0])
    centroids = [[0 for _ in range(D)] for _ in range(k)]
    count = [0] * k 
    for ip, p in enumerate(points):
        ic = assignments[ip]
        for j in range(D):
            centroids[ic][j] += p[j]
        count[ic] += 1
    
    for ic in range(k):
        if count[ic] > 0:
            for j in range(D):
                centroids[ic][j] /= count[ic]
    
    return centroids