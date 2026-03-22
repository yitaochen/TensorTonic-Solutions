def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        num = 0
        mu = 0
        for j in range(n):
            if matrix[i][j] != 0:
                num += 1
                mu += matrix[i][j]
        if num == 0:
            matrix[i] = [0.0] * n
        else:
            mu /= num 
            matrix[i] = [v-mu if v!=0 else 0.0 for v in matrix[i]]

    return matrix 