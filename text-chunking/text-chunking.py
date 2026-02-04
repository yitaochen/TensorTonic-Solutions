def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    ans = []
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        if i + chunk_size >= len(tokens):
            break
        ans.append(tokens[i:i+chunk_size])
    ans.append(tokens[i:])

    return ans 