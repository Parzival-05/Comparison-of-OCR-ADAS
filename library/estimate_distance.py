FOCAL_LENGTH = 440  # 2.2mm in pixels
H_CAM = 1.2  # height of the camera
H_OBJ = 5.5  # height of the road sign
HOR = 585  # horizon


def estimate_distance(hor_obj):
    return round(FOCAL_LENGTH * (H_OBJ - H_CAM) / -(hor_obj - HOR), 2)
