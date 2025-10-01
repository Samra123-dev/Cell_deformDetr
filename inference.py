import argparse
import torch
import cv2
import numpy as np
from models import build_model
import util.misc as utils
from main import get_args_parser
from PIL import Image
import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

def main(args):
    # build model
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().to(args.device)

    # load image
    transform = get_transform()
    img = Image.open(args.image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(args.device)

    # inference
    outputs = model(img_tensor)

    # get predictions
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    bboxes = outputs["pred_boxes"][0, keep].cpu()
    probas = probas[keep].cpu()

    # draw results
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    for p, (cx, cy, bw, bh) in zip(probas, bboxes):
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        x0, y0, x1, y1 = int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2)
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img_cv)
    print("Saved result.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Deformable DETR inference", parents=[get_args_parser()])
    parser.add_argument("--image_path", required=True, help="path to input image")
    parser.add_argument("--checkpoint", required=True, help="path to model checkpoint")
    args = parser.parse_args()
    main(args)

