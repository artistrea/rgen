import cv2 as cv
import numpy as np


# def scenario_img2yaml(
#     *,
#     img_path=None,
#     bitmap: np.ndarray =None
# ) -> dict:
#     """
#     This functcion converts image walls to a dictionary following
#     the format here defined for scenes
#     """
#     if bitmap is None and img_path is not None:
#         # numpy array (N, M, 3)
#         bitmap = cv2.imread(img_path)

#     rows, cols, chann = bitmap.shape
#     for i in range(rows):
#         for j in range(cols):
#             if bitmap[i][j] == np.zeros(chann):
                

#     return scenario
