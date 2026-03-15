def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    h, w = len(kernel), len(kernel[0])
    H, W = len(image), len(image[0])
    H_out, W_out = (H + 2*padding - h)//stride + 1, (W + 2*padding - w)//stride + 1
    # pad the image
    padded = [[0] * (W + 2*padding) for _ in range(H + 2*padding)]
    for i in range(H):
        for j in range(W):
            padded[i+padding][j+padding] = image[i][j]
    # build the convolution result
    ans = [[0] * W_out for _ in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            ans[i][j] = sum(padded[i*stride+m][j*stride+n]*kernel[m][n] for m in range(h) for n in range(w))

    return ans 
    