def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    n = len(rater1)
    num_classes = max(rater1 + rater2) + 1
    cnt1 = [0] * num_classes
    cnt2 = [0] * num_classes
    num_agreements = 0
    for i in range(n):
        if rater1[i] == rater2[i]:
            num_agreements += 1
        cnt1[rater1[i]] += 1
        cnt2[rater2[i]] += 1
    pe = sum(cnt1[i]*cnt2[i]/(n*n) for i in range(num_classes))
    if pe == 1:
        return 1.0
    else:
        return (num_agreements/n - pe) / (1 - pe)