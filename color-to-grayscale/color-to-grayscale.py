def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    h, w = len(image), len(image[0])
    R=0.299
    G=0.587
    B=0.114
    ans = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            ans[i][j] = R * image[i][j][0] + G * image[i][j][1] + B * image[i][j][2]

    return ans 