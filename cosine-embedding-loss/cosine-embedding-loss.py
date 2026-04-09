def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    def norm(nums):
        return sum(num**2 for num in nums)**0.5
    def dot_product(v1, v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    cos = dot_product(x1, x2) / (norm(x1) * norm(x2))

    return 1 - cos if label == 1 else max(0, cos - margin)