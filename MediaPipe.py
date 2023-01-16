import cv2
import numpy as np
import mediapipe as mp

# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)


def remove_background(file_slika_osebe):
    # load image
    slika_osebe = cv2.imread(f"slika osebe/{file_slika_osebe}")
    slika_osebe_rgb = cv2.cvtColor(slika_osebe, cv2.COLOR_BGR2RGB)

    # get the result
    results = selfie_segmentation.process(slika_osebe_rgb)

    # it returns true or false where the condition applies in the mask
    mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5

    # bg_image = np.zeros(slika_osebe.shape, dtype=np.uint8)
    # final_image = np.where(mask, slika_osebe, bg_image)

    return mask
