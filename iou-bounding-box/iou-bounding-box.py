def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    area_a = (box_a[3] - box_a[1]) * (box_a[2] - box_a[0])
    area_b = (box_b[3] - box_b[1]) * (box_b[2] - box_b[0])
    intx_1 = max(box_a[0], box_b[0])
    intx_2 = min(box_a[2], box_b[2])
    inty_1 = max(box_a[1], box_b[1])
    inty_2 = min(box_a[3], box_b[3])
    area_int = (inty_2 - inty_1) * (intx_2 - intx_1) if inty_2 > inty_1 and intx_2 > intx_1 else 0.0
    area_union = area_a + area_b - area_int

    return area_int / area_union if area_union > 0 else 0.0