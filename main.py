import argparse
import cv2
import time

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

def process_image(_frame, device, seg_model):

    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    _frame = cv2.resize(_frame, tuple(cfg.IMAGE.SIZE))
    img_display = _frame

    img = Image.fromarray(_frame).convert('RGB')
    seg_prediction = segmentation.predict_one(seg_model, img, device, cfg)

    model_name = "swaftnet"
    convert = "segformer" not in model_name

    colorized = colorize.colorize(seg_prediction, palette=cfg.SEGMENTATION.PALETTE, convert=convert)
    colorized_seg_pred = np.asarray(colorized.copy())

    walkable_area_mask = extract_walkable_area(colorized_seg_pred, cfg)
    walkable_area_mask_resized = cv2.resize(walkable_area_mask.astype(np.uint8), (_frame.shape[1], _frame.shape[0]))
    # divide the resulting matrix in 4 equal quadrants and multiply by 1.5
    # print(walkable_area_mask_resized)


    # Convert the binary mask to a 3-channel image to use for blending
    walkable_area_mask_3ch = cv2.cvtColor(walkable_area_mask_resized, cv2.COLOR_GRAY2BGR)

    # Blend the mask with the original frame
    # Note: Adjust blending factor as needed
    # blended_frame = cv2.addWeighted(_frame, 0.7, walkable_area_mask_3ch * 255, 0.3, 0)

    # return blended_frame

    blended = colorize.blend(img, colorized, cfg.SEGMENTATION.ALPHA)
    img_display = display(_frame, colorized, blended, cfg.DRONE.DISPLAY_IMAGE)
    img_display = np.asarray(img_display)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

    return img_display

def test_on_image(device,seg_model):
    image_paths = ["2.jpeg"]
    for image_path in image_paths:
        _frame = cv2.imread(image_path)
        if _frame is None:
            raise FileNotFoundError(f"Image {image_path} not found.")
        # # ret_val, _frame = video.read()
        # if not ret_val:
        #     break
        frame =  process_image(_frame, device, seg_model)

        
        cv2.imshow("Walkable Area Mask", frame)
        # cv2.imshow("Blended Image", img_display)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        #     break

        # Wait for a key press to close the displayed windows
        cv2.destroyAllWindows()


def test_on_recording(device,seg_model):
    video = cv2.VideoCapture('./assets/indoors.MOV')

    output_path = './assets/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to encode the video
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Capture the frames per second of the input video
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        ret_val, _frame = video.read()
        if not ret_val:
            break
        # cv2.imshow("Original Video Frame", _frame)

        # Uncomment and implement process_image function as needed
        frame = process_image(_frame, device, seg_model)

        out.write(frame)
        # cv2.imshow("Processed Frame", frame)


        # # Display the frame (optional, can be removed for faster processing)
        # cv2.imshow("Processed Frame", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        #     break

        # This line is only needed if you're displaying another window
        # cv2.imshow("Blended Image", img_display)

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        #     break
    video.release()
    out.release()  # Don't forget to release the VideoWriter
    cv2.destroyAllWindows()


def test_on_video(device,seg_model):
    video = cv2.VideoCapture(0)

    num_frame = 0
    start_time = time.time()  # Start time for FPS calculation
    while True:
            ret_val, _frame = video.read()
            if not ret_val:
                break

            current_time = time.time()
            fps = num_frame / (current_time - start_time)


            frame =  process_image(_frame, device, seg_model)

            # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Walkable Area Mask", frame)
            # cv2.imshow("Blended Image", img_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

            # Wait for a key press to close the displayed windows
            cv2.destroyAllWindows()

            num_frame += 1
    video.release()
    cv2.destroyAllWindows()

def main():


    device = torch.device("cpu")
    seg_model = segmentation.build_seg_model("mobilenet_v3_small")()

    # test_on_image(device,seg_model)
    test_on_video(device,seg_model)
    # test_on_recording(device,seg_model)


if __name__ == "__main__":
    main()
