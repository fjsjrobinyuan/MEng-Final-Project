import os
import random
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile

# ----------------------------
# 1. DATASET & UTILITIES
# ----------------------------
def download_yolo_dataset(dataset_url="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
                          dest_folder="data"):
    """
    Downloads and extracts a small YOLO-format dataset (default: COCO128).
    The dataset has images and labels already in the right structure:
      data/images/
      data/labels/
    """
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "dataset.zip")

    if not os.path.exists(os.path.join(dest_folder, "images")):
        print(f"📦 Downloading dataset from {dataset_url} ...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("✅ Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)
        print("🎉 Dataset ready!")
    else:
        print("✅ Dataset already exists, skipping download.")


class YOLODataset(Dataset):
    """
    Expects images and label files in YOLO format.
    image_dir: folder with .jpg/.png images
    label_dir: folder with .txt labels with lines: class x_center y_center width height (normalized)
    img_size: width = height (square)
    """
    def __init__(self, image_dir, label_dir, img_size=416, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).astype(np.float32) / 255.0  # normalize to [0,1]
        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1)

        # parse labels
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = parts
                    targets.append([float(cls), float(x), float(y), float(w), float(h)])
        targets = torch.tensor(targets)  # shape (num_boxes, 5)
        return img, targets


def xywh_to_xyxy(box):
    # box: (..., 4) [x_center, y_center, w, h]
    x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between box1 and box2, boxes in (x_center,y_center,w,h) form or (x1,y1,x2,y2).
    Both must be same shape or broadcastable.
    """
    # If in xywh, convert to xyxy
    if box1.shape[-1] == 4:
        box1 = xywh_to_xyxy(box1)
    if box2.shape[-1] == 4:
        box2 = xywh_to_xyxy(box2)

    # (…, 4) with (x1,y1,x2,y2)
    x1 = torch.max(box1[...,0], box2[...,0])
    y1 = torch.max(box1[...,1], box2[...,1])
    x2 = torch.min(box1[...,2], box2[...,2])
    y2 = torch.min(box1[...,3], box2[...,3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (box1[...,2] - box1[...,0]).clamp(min=0) * (box1[...,3] - box1[...,1]).clamp(min=0)
    area2 = (box2[...,2] - box2[...,0]).clamp(min=0) * (box2[...,3] - box2[...,1]).clamp(min=0)
    union = area1 + area2 - inter + eps

    return inter / union


# ----------------------------
# 2. MODEL
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLO(nn.Module):
    def __init__(self, num_classes, anchors, img_size=416):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors  # list of (w, h) in *grid* units
        self.num_anchors = len(anchors)
        self.img_size = img_size

        # backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, 3),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, 3),
            nn.MaxPool2d(2),
            ConvBlock(256, 512, 3),
            nn.MaxPool2d(2),
            ConvBlock(512, 1024, 3),
        )

        # head
        out_c = self.num_anchors * (5 + self.num_classes)
        self.head = nn.Conv2d(1024, out_c, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.backbone(x)
        pred = self.head(feat)
        A = self.num_anchors
        C = 5 + self.num_classes
        gh, gw = pred.shape[2], pred.shape[3]
        pred = pred.view(B, A, C, gh, gw)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        return pred  # shape (B, A, gh, gw, 5+num_classes)


# ----------------------------
# 3. LOSS + TARGET ASSIGNMENT
# ----------------------------

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
        super().__init__()
        self.anchors = torch.tensor(anchors)  # shape (A, 2), widths & heights in *grid cell units*
        self.num_anchors = anchors.__len__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def forward(self, preds, targets):
        """
        preds: (B, A, gh, gw, 5 + C)
        targets: list of target boxes per-image; not per-grid tensor
        We convert targets into a properly shaped target tensor internally.
        """
        B, A, gh, gw, _ = preds.shape
        device = preds.device

        # Prepare target tensor: (B, A, gh, gw, 5 + C)
        t = torch.zeros((B, A, gh, gw, 5 + self.num_classes), device=device)

        # Fill in target tensor by assigning each ground truth box to an anchor+grid cell
        for b in range(B):
            for gt in targets[b]:
                cls, x, y, w, h = gt
                # convert normalized xy to grid-cell units
                gx = x * gw
                gy = y * gh
                gi = int(gx)
                gj = int(gy)
                # width / height in grid units
                gw_box = w * gw
                gh_box = h * gh
                # compute which anchor fits best (by IoU between box and each anchor)
                box = torch.tensor([gx - gi, gy - gj, gw_box, gh_box], device=device)
                anchors = self.anchors.to(device)  # move anchors to GPU if needed
                # compute IoU with anchors (anchors are like boxes centered at 0,0)
                anchor_boxes = torch.cat([torch.zeros((A,2), device=device),
                                        anchors], dim=1)  # shape (A,4) as (0,0,aw,ah)
                # build GT box repeated
                box_rep = box.unsqueeze(0).repeat(A,1)
                ious = bbox_iou(box_rep, anchor_boxes)
                best_n = torch.argmax(ious).item()

                # set objectness = 1
                t[b, best_n, gj, gi, 4] = 1.0
                # set relative x,y (offset within cell)
                t[b, best_n, gj, gi, 0] = gx - gi
                t[b, best_n, gj, gi, 1] = gy - gj
                # width/height relative (we can use log space in advanced versions)
                t[b, best_n, gj, gi, 2] = gw_box
                t[b, best_n, gj, gi, 3] = gh_box
                # one-hot class
                t[b, best_n, gj, gi, 5 + int(cls)] = 1.0

        # Now split preds
        pred_xy = preds[..., 0:2]
        pred_wh = preds[..., 2:4]
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]

        t_xy = t[..., 0:2]
        t_wh = t[..., 2:4]
        t_obj = t[..., 4]
        t_cls = t[..., 5:]

        # Loss components

        # Box loss (MSE) only where object exists
        loss_box = self.lambda_box * ((t_obj * ((pred_xy - t_xy).pow(2).sum(-1) + (pred_wh - t_wh).pow(2).sum(-1))).sum())

        # Objectness loss
        loss_obj = self.lambda_obj * ((t_obj * (pred_obj - 1).pow(2)).sum())
        # No object
        loss_noobj = self.lambda_noobj * (((1 - t_obj) * (pred_obj).pow(2)).sum())

        # Classification loss (MSE over classes) where object is
        loss_cls = self.lambda_cls * ((t_obj.unsqueeze(-1) * (pred_cls - t_cls).pow(2)).sum())

        loss = loss_box + loss_obj + loss_noobj + loss_cls
        return loss


# ----------------------------
# 4. NON-MAXIMUM SUPPRESSION (for inference)
# ----------------------------

def non_max_suppression(prediction, conf_threshold=0.5, iou_threshold=0.5):
    """
    prediction: list of per-image predictions (B, A, gh, gw, 5 + C)
    Returns detections: list of (num_dets, 6) [x1, y1, x2, y2, confidence, class]
    """
    output = []
    for pred in prediction:
        # pred: (A, gh, gw, 5 + C)
        A, gh, gw, nc = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        # flatten to (A * gh * gw, 5 + C)
        flat = pred.view(-1, 5 + (nc - 5))
        boxes = flat[:, :4]
        objness = flat[:, 4]
        class_scores = flat[:, 5:]
        class_conf, class_pred = torch.max(class_scores, dim=1)

        scores = objness * class_conf
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = class_pred[mask]

        if boxes.shape[0] == 0:
            output.append(torch.zeros((0,6)))
            continue

        # convert box format to xyxy (currently xywh-like)
        boxes_xyxy = xywh_to_xyxy(boxes)

        # sort by score
        scores, order = scores.sort(descending=True)
        boxes_xyxy = boxes_xyxy[order]
        classes = classes[order]

        keep = []
        while boxes_xyxy.size(0) > 0:
            i = 0
            best_box = boxes_xyxy[i].unsqueeze(0)  # shape (1,4)
            best_score = scores[i]
            best_cls = classes[i]
            keep.append([*best_box[0].tolist(), best_score.item(), best_cls.item()])

            if boxes_xyxy.size(0) == 1:
                break
            rest_boxes = boxes_xyxy[1:]
            rest_scores = scores[1:]
            rest_classes = classes[1:]

            ious = bbox_iou(best_box.repeat(rest_boxes.size(0),1), rest_boxes)
            mask_keep = ious < iou_threshold

            boxes_xyxy = rest_boxes[mask_keep]
            scores = rest_scores[mask_keep]
            classes = rest_classes[mask_keep]

        output.append(torch.tensor(keep))
    return output


# ----------------------------
# 5. TRAIN / INFER DRIVER
# ----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        imgs, targets = batch  # unpack the tuple

        # Stack image tensors into a single batch tensor
        imgs = torch.stack(imgs).to(device)

        # Move targets to device (list of tensors)
        targets = [t.to(device) for t in targets]

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, epochs, optimizer, criterion, device, base_lr=1e-4):
    os.makedirs("weight", exist_ok=True)

    # ----------------------------
    # OneCycleLR Scheduler setup
    # ----------------------------
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=base_lr * 10,              # peak LR (10× base is usually good)
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,                    # 30% of total steps used for warm-up
        anneal_strategy='cos',            # cosine decay (smoother than linear)
        div_factor=25.0,                  # initial LR = max_lr / 25
        final_div_factor=1e4,             # final LR = max_lr / 1e4
        three_phase=False,                # standard two-phase curve
    )

    prev_loss = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets in train_loader:
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # advance OneCycleLR *per batch*, not per epoch

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {ep}/{epochs}: loss = {avg_loss:.4f}, lr = {curr_lr:.8f}")

        torch.save(model.state_dict(), f"weight/yolo_epoch{ep}.pth")
        prev_loss = avg_loss




def inference(model, img, device):
    # img: single image tensor (C, H, W), normalized [0,1]
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))  # shape (1, A, gh, gw, 5+C)
        det = non_max_suppression(pred.cpu(), conf_threshold=0.4, iou_threshold=0.5)
    return det[0]


# ----------------------------
# 6. MAIN: setup and run
# ----------------------------
if __name__ == "__main__":
    # basic config
    IMG_SIZE = 416
    BATCH_SIZE = 16
    NUM_CLASSES = 80  # COCO128 has 80 classes
    EPOCHS = 60
    LR = 1e-4
    ANCHORS = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    # ----------------------------
    # DOWNLOAD DATASET
    # ----------------------------
    download_yolo_dataset()
    IMAGE_DIR = "data/coco128/images/train2017"
    LABEL_DIR = "data/coco128/labels/train2017"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = YOLODataset(IMAGE_DIR, LABEL_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    model = YOLO(num_classes=NUM_CLASSES, anchors=ANCHORS, img_size=IMG_SIZE).to(device)
    criterion = YOLOLoss(anchors=ANCHORS, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, dataloader, EPOCHS, optimizer, criterion, device)

    # (Optional) Test inference on a sample image
    # img, _ = dataset[0]
    # dets = inference(model, img, device)
    # print("Detections:", dets)
