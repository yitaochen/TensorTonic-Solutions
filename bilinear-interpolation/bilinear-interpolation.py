def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here
    h, w = len(image), len(image[0])
    new_image = [[0] * new_w for _ in range(new_h)]
    for i in range(new_h):
        for j in range(new_w):
            src_y = i * (h - 1) / (new_h - 1) if new_h - 1 > 0 else 0
            src_x = j * (w - 1) / (new_w - 1) if new_w - 1 > 0 else 0
            y0 = int(src_y)
            dy = src_y - y0 
            y1 = min(y0+1, h-1)
            x0 = int(src_x)
            dx = src_x - x0 
            x1 = min(x0+1, w-1)
            new_image[i][j] = image[y0][x0]*(1-dy)*(1-dx) \
                            + image[y1][x0]*dy*(1-dx) \
                            + image[y0][x1]*(1-dy)*dx \
                            + image[y1][x1]*dy*dx

    return new_image