import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import PIL.Image

#returns a set of labels
@smart_inference_mode()
def predict(image_path, weights='yolov9-c.pt', imgsz=640, conf_thres=0.1, iou_thres=0.45):
    # Initialize
    device = 'cpu'
    model = DetectMultiBackend(weights='yolov9-c.pt', fp16=False, data='data/coco.yaml')
    stride, names, pt = model.stride, model.names, model.pt

    # Load image
    image = np.array(PIL.Image.open(image_path))
    img = letterbox(image, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)

    # Apply NMS
    pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)
    predictions = set()
    for det in pred:  # per image
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                predictions.add(names[int(cls)])

    return predictions