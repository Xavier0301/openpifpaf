import numpy as np

BBOX_KEYPOINTS = [
    'top_left',            # 1
    'top_right',        # 2
    'bottom_right',       # 3
    'bottom_left',        # 4
    'center',       # 5
]

BBOX_SKELETON = [
    [1,5], [2,5], [3,5], [4,5]
]

DENSER_BBOX_SKELETON = [
    [1, 2], [2, 3], [3, 4], [4, 1], [1,5], [2,5], [3,5], [4,5]
]

DENSER_BBOX_CONNECTIONS = [
    c
    for c in DENSER_BBOX_SKELETON
    if c not in BBOX_SKELETON
]

BBOX_SIGMAS = [
    0.1,  # top_left
    0.1,  # top_right
    0.1,  # bottom_right
    0.1,  # bottom_left
    0.1,  # center
]

BBOX_UPRIGHT_POSE = np.array([
    [-1.0, -1.0, 2.0],# 'nose',            # 1
    [-1, 1, 2.0],  # 'left_eye',        # 2
    [1, 1, 2.0],  # 'right_eye',       # 3
    [1, -1, 2.0],  # 'left_ear',        # 4
    [0.0, 0.0, 2.0],  # 'right_ear',       # 5
])

BBOX_HFLIP = {
    'top_left': 'top_right',
    'top_right': 'top_left',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
    'center': 'center',
}
