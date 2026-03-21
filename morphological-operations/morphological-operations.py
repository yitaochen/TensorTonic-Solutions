def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    # Write code here
    H, W = len(image), len(image[0])
    H_k, W_k = len(kernel), len(kernel[0])
    padded = [[0]*(W+W_k//2*2) for _ in range(H+H_k//2*2)]
    for i in range(H):
        for j in range(W):
            padded[i+H_k//2][j+W_k//2] = image[i][j]
    output = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            early_stop = False
            for m in range(H_k):
                if early_stop:
                    break
                for n in range(W_k):
                    if operation == "erode":
                        if kernel[m][n] == 1 and padded[i+m][j+n] == 0:
                            early_stop = True 
                            break 
                    else:
                        if kernel[m][n] == 1 and padded[i+m][j+n] == 1:
                            early_stop = True 
                            break 
            if operation == "erode":
                if not early_stop:
                    output[i][j] = 1
            else:
                if early_stop:
                    output[i][j] = 1

    return output
                            