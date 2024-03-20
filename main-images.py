import argparse
import cv2
import numpy as np
from PIL import Image
import torch

from models import segmentation, traffic_light_classification
from config import cfg, update_config
from utils import colorize
from utils.utils import (
    create_logger,
    extract_walkable_area,
    display,
    extract_traffic_light,
)

def reduce_green_channel(image, reduction_factor=0.7):
    image[:, :, 1] = image[:, :, 1] * reduction_factor
    return image

def apply_sharp_edges_filter(image):
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1,  5, -1],
                                  [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

def process_image(image_path, device, seg_model):
    # Load the image
    _frame = cv2.imread(image_path)
    if _frame is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    _frame = cv2.resize(_frame, tuple(cfg.IMAGE.SIZE))
    # _frame = reduce_green_channel(_frame)
    _frame = apply_sharp_edges_filter(_frame)

    img = Image.fromarray(_frame).convert('RGB')
    img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to(device)

    seg_prediction = segmentation.predict_one(seg_model, img, device, cfg)

    model_name = "mobilenet_v3_small"
    convert = "segformer" not in model_name
    colorized = colorize.colorize(seg_prediction, palette=cfg.SEGMENTATION.PALETTE, convert=convert)
    colorized_seg_pred = np.asarray(colorized.copy())

    walkable_area_mask = extract_walkable_area(colorized_seg_pred, cfg)

    blended = colorize.blend(img, colorized, cfg.SEGMENTATION.ALPHA)
    img_display = display(_frame, colorized, blended, cfg.DRONE.DISPLAY_IMAGE)
    img_display = np.asarray(img_display)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

    # cv2.imshow("Segmentation Colorized", colorized_seg_pred)
    cv2.imshow("Walkable Area Mask", walkable_area_mask)
    # cv2.imshow("Blended Image", img_display)

    # Wait for a key press to close the displayed windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    device = torch.device("cpu")
    seg_model = segmentation.build_seg_model("mobilenet_v3_small")()

    image_paths = ["2.jpeg"]
    for image_path in image_paths:
        process_image(image_path, device, seg_model)

if __name__ == "__main__":
    main()
