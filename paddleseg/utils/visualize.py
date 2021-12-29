# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
from PIL import Image as PILImage

def reseizeToMaxSide(img, maxSide = 480):
    scaleVal = maxSide / max(img.shape[0:2])
    return cv2.resize(img, (round(img.shape[1] * scaleVal), round(img.shape[0] * scaleVal)))

def padToSquare(image, fill=0):
    h, w = image.shape[0:2]
    max_wh = max([w, h])
    wp = int((max_wh - w) / 2)
    hp = int((max_wh - h) / 2)
    hp2 = hp + (max_wh - (h + hp * 2))
    wp2 = wp + (max_wh - (w + wp * 2))
    return cv2.copyMakeBorder(image, hp, hp2, wp, wp2, cv2.BORDER_REFLECT_101)

def PadAndResizeTo(img, input_size):
    return padToSquare(reseizeToMaxSide(img, input_size))

def UnPadAndResizeTo(img, w, h = None):
    if h == None:
        h = w
    max_wh = max([w, h])
    radio = img.shape[0] / max_wh 
    tmpW, tmpH = radio * w, radio * h
    tmpMaxWH = max([tmpW, tmpH])
    ch = round((tmpMaxWH - tmpH) / 2)
    cw = round((tmpMaxWH - tmpW) / 2)
    img = img[ch : img.shape[0] - ch, cw : img.shape[0] - cw]
    img = cv2.resize(img, (w, h))
    return img

def visualize_single_class(image, pred : np.array):
    mask = np.clip(pred, 0, 1)
    mask *= 255
    mask = mask.astype('uint8')
    b, g, r = cv2.split(image)
    reImage = cv2.merge((b, g, r, mask))
    return reImage, mask

def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.
    
    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map
