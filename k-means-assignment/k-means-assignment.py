def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    N = len(points)
    D = len(points[0])
    assignment = [-1] * N 
    for ip, p in enumerate(points):
        index = -1
        MIN = float('inf')
        for ic, c in enumerate(centroids):
            distance = sum((p[d]-c[d])**2 for d in range(D))
            if distance < MIN:
                index = ic
                MIN = distance
        assignment[ip] = index 
    
    return assignment

