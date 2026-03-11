import math 

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    if not candidate:
        return 0.0
    c, r = len(candidate), len(reference)
    BP = 1 if c >= r else math.exp(1-r/c)
    p = []
    for n in range(1, max_n+1):
        cnt_c = {}
        cnt_r = {}
        for i in range(c-n+1):
            ngram_c = " ".join(candidate[i:i+n])
            cnt_c[ngram_c] = cnt_c.get(ngram_c, 0) + 1
        for i in range(r-n+1):
            ngram_r = " ".join(reference[i:i+n])
            cnt_r[ngram_r] = cnt_r.get(ngram_r, 0) + 1
        denom = sum(v for k, v in cnt_c.items())
        if denom > 0:
            pn = sum(min(v, cnt_r.get(k, 0)) for k, v in cnt_c.items()) / denom
        else:
            pn = 0 
        if pn == 0:
            return 0.0
        p.append(pn)
    # print(p)
    return BP * math.exp(sum(math.log(x) for x in p) / max_n)
        