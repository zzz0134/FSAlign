# -*- coding: utf-8 -*-
"""
fm_single_allentities_parts.py
------------------------------
Single-image pipeline that:
  1) Detects ALL visible entities via a hybrid detector union:
       - YOLO (COCO, fast & strong)
       - (optional) GroundingDINO (open-vocabulary, expands beyond COCO)
       - (optional) DeepLab (semantic 'stuff' regions like tv/sofa/table; approximate via VOC)
  2) For *every detected entity*, attaches PARTS:
       A. (optional) Open-vocabulary PART prompts (GroundingDINO) per class -> semantic parts (wheel, ear, screen...)
       B. Generic unsupervised parts fallback for any class (ORB clusters; optional SLIC superpixels if skimage available)
       C. (optional) Human parts via MediaPipe (hands/face->hair) merged automatically
  3) Builds L0/L1/L2 panels and a type-aware fractal manifold.
  4) Saves a JSON dump including all parts and attributes per entity.

Install (minimum):
    pip install ultralytics opencv-python matplotlib numpy pillow scikit-learn spacy

Optional (recommended):
    pip install groundingdino-py torch torchvision   # open-vocabulary parts/detection
    pip install mediapipe                             # hands/face
    pip install scikit-image                          # SLIC superpixels for nicer generic parts
    python -m spacy download en_core_web_sm

Usage:
    python fm_single_allentities_parts.py \
      --img /path/to/image.jpg \
      --caption "Your caption" \
      --out ./fm_out \
      --yolo_model yolov8x.pt --conf 0.25 --imgsz 1024 \
      --ov_on --ov_cfg GroundingDINO_SwinT_OGC.py --ov_weights groundingdino_swint_ogc.pth \
      --seg_on \
      --parts 3
"""
import os, json, argparse, re, warnings, tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

# ---------- Optional imports ----------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

# GroundingDINO (open-vocabulary)
try:
    from groundingdino.util.inference import load_model, load_image, predict
    _G_DINO_OK = True
except Exception:
    _G_DINO_OK = False

# TorchVision DeepLab (semantic segmentation for 'stuff')
try:
    import torch
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet101
    _DEEPLAB_OK = True
except Exception:
    _DEEPLAB_OK = False

# scikit-image SLIC
try:
    from skimage.segmentation import slic
    _SKIMAGE_OK = True
except Exception:
    _SKIMAGE_OK = False

# MediaPipe (optional parts like hands/hair if visible)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_face  = mp.solutions.face_detection
except Exception:
    mp = None; mp_hands = None; mp_face = None

# ----------------------------
# Data structures and helpers
# ----------------------------
@dataclass
class BBox:
    x0:int; y0:int; x1:int; y1:int
    score:float=1.0; cls:str=""
    def as_tuple(self): return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))
    def center(self): return ((self.x0+self.x1)/2.0, (self.y0+self.y1)/2.0)
    def area(self): return max(0, self.x1-self.x0)*max(0, self.y1-self.y0)
    def intersect(self, other:'BBox'):
        x0, y0 = max(self.x0,other.x0), max(self.y0,other.y0)
        x1, y1 = min(self.x1,other.x1), min(self.y1,other.y1)
        return max(0,x1-x0)*max(0,y1-y0)
    def iou(self, other:'BBox'):
        inter = self.intersect(other); u = self.area()+other.area()-inter
        return inter/u if u>0 else 0.0
    def contains_point(self, x:int, y:int):
        return (self.x0 <= x <= self.x1) and (self.y0 <= y <= self.y1)

def nms_boxes(boxes: List[BBox], iou_th=0.5, same_class_only=False) -> List[BBox]:
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b:(b.score), reverse=True)
    kept=[]
    while boxes:
        b = boxes.pop(0); kept.append(b)
        if same_class_only:
            boxes = [x for x in boxes if b.cls != x.cls or b.iou(x) < iou_th]
        else:
            boxes = [x for x in boxes if b.iou(x) < iou_th]
    return kept

def iou(a: BBox, b: BBox) -> float: return a.iou(b)

def merge_union(batches: List[List[BBox]], iou_th=0.6) -> List[BBox]:
    all_boxes = []
    for batch in batches:
        all_boxes.extend(batch)
    # class-aware merge
    all_boxes = sorted(all_boxes, key=lambda b:b.score, reverse=True)
    out=[]
    for b in all_boxes:
        if all(iou(b, x) < iou_th or (b.cls != x.cls) for x in out):
            out.append(b)
    return out

def draw_boxes(ax, img_bgr: np.ndarray, boxes: Dict[str,BBox], title:str):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    for name,b in boxes.items():
        x0,y0,x1,y1 = b.as_tuple()
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
        ax.text(x0, max(0,y0-4), name, fontsize=9, va='bottom', color="white",
                bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.5))
    ax.set_title(title); ax.axis('off')

def save_text_panel(lines: List[str], title:str, out_path:str, figsize=(8,3.2)):
    fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
    ax.axis('off'); ax.set_title(title); y=0.9
    for line in lines:
        ax.text(0.02, y, line, fontsize=12, va='top', wrap=True); y -= 0.15
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)

def crop(img_bgr: np.ndarray, box: BBox) -> np.ndarray:
    x0,y0,x1,y1 = box.as_tuple()
    x0=max(0,x0); y0=max(0,y0); x1=min(img_bgr.shape[1]-1,x1); y1=min(img_bgr.shape[0]-1,y1)
    if x1<=x0 or y1<=y0: return np.zeros((0,0,3), dtype=np.uint8)
    return img_bgr[y0:y1, x0:x1].copy()

def dominant_color(bgr: np.ndarray, k:int=3) -> Tuple[int,int,int]:
    if bgr.size==0: return (0,0,0)
    data = bgr.reshape(-1,3).astype(np.float32)
    k = min(k, max(1, data.shape[0]//200))  # avoid tiny patches overclustering
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(data)
    centers = km.cluster_centers_.astype(int); counts = np.bincount(km.labels_)
    idx = int(np.argmax(counts)); c = centers[idx]
    return (int(c[0]), int(c[1]), int(c[2]))

def edge_energy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))

# ----------------------------
# Detection backends
# ----------------------------
class YOLODetector:
    def __init__(self, model_path:str, conf:float, imgsz:int):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, img_bgr: np.ndarray) -> List[BBox]:
        r = self.model.predict(img_bgr[..., ::-1], imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        boxes=[]
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x0,y0,x1,y1 = map(int, b); cls = r.names[int(c)]
            boxes.append(BBox(x0,y0,x1,y1, float(s), cls))
        return nms_boxes(boxes, iou_th=0.6, same_class_only=True)

class OVBDetector:
    """Open-vocabulary detection via GroundingDINO (optional)."""
    def __init__(self, cfg_path:str, wt_path:str, box_threshold:float=0.25, text_threshold:float=0.25):
        if not _G_DINO_OK:
            raise RuntimeError("GroundingDINO not installed. pip install groundingdino-py")
        self.model = load_model(cfg_path, wt_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, img_path:str, text_prompt:str) -> List[BBox]:
        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=self.model, image=image, caption=text_prompt,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        H, W = image_source.shape[:2]
        out=[]
        for b, p, s in zip(boxes, phrases, logits):
            cx,cy,bw,bh = b  # normalized cx,cy,w,h
            x0 = int((cx - bw/2) * W); y0 = int((cy - bh/2) * H)
            x1 = int((cx + bw/2) * W); y1 = int((cy + bh/2) * H)
            out.append(BBox(x0,y0,x1,y1, float(s), p))
        return nms_boxes(out, iou_th=0.6, same_class_only=True)

class DeepLabStuff:
    """Semantic segmentation (stuff-like regions) using torchvision's DeepLabV3; adds sky/grass/road/building-like entities."""
    VOC21 = [
        "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
        "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
    ]
    MAP_TO_STUFF = {"diningtable":"table","sofa":"couch","tvmonitor":"tv","motorbike":"motorbike"}
    def __init__(self):
        if not _DEEPLAB_OK:
            raise RuntimeError("torchvision DeepLab not available")
        self.model = deeplabv3_resnet101(weights="DEFAULT").eval()

    @torch.no_grad()
    def segment(self, img_bgr: np.ndarray, min_area:int=800) -> List[BBox]:
        H,W = img_bgr.shape[:2]
        to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inp = to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        out = self.model(inp)["out"][0]  # [C,H,W]
        mask = out.argmax(0).cpu().numpy().astype(np.uint8)
        boxes=[]
        for cid in np.unique(mask):
            if cid == 0:  # background
                continue
            cls = self.VOC21[cid]; cls = self.MAP_TO_STUFF.get(cls, cls)
            ys, xs = np.where(mask == cid)
            if ys.size < 5: continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if (x1-x0)*(y1-y0) < min_area: continue
            boxes.append(BBox(x0,y0,x1,y1, 0.35, cls))
        return nms_boxes(boxes, iou_th=0.6, same_class_only=True)

# ----------------------------
# Caption parsing & noun prompts
# ----------------------------
LEXICON = {
    "man":"person","men":"person","woman":"person","women":"person","guy":"person","boy":"person","girl":"person","people":"person",
    "dog":"dog","cat":"cat","horse":"horse","cow":"cow","sheep":"sheep","bird":"bird",
    "car":"car","truck":"truck","bus":"bus","bicycle":"bicycle","motorcycle":"motorbike","motorbike":"motorbike","train":"train","boat":"boat","airplane":"aeroplane","plane":"aeroplane",
    "bench":"bench","chair":"chair","sofa":"couch","couch":"couch","bed":"bed","table":"dining table","tv":"tv","monitor":"tv","laptop":"laptop",
    "keyboard":"keyboard","mouse":"mouse","cell":"cell phone","phone":"cell phone","remote":"remote",
    "bottle":"bottle","cup":"cup","bowl":"bowl","wine":"wine glass","glass":"wine glass",
    "backpack":"backpack","handbag":"handbag","umbrella":"umbrella","tie":"tie",
    "frisbee":"frisbee","skis":"skis","snowboard":"snowboard","kite":"kite",
    "baseball":"baseball bat","bat":"baseball bat","glove":"baseball glove","tennis":"tennis racket","racket":"tennis racket",
    "book":"book","clock":"clock","vase":"vase","scissors":"scissors","toilet":"toilet",
    # background/stuff hints
    "fence":"fence","house":"house","building":"building","porch":"porch","window":"window","door":"door",
    "tree":"tree","grass":"grass","yard":"yard","garden":"garden","sky":"sky","road":"road","street":"street","ground":"ground"
}

def nouns_from_caption(caption:str) -> List[str]:
    toks = re.findall(r"[A-Za-z]+", caption.lower())
    nouns = set()
    if _SPACY_OK:
        try:
            nlp = spacy.load("en_core_web_sm"); doc = nlp(caption)
            nouns |= {t.lemma_.lower() for t in doc if t.pos_ in ("NOUN","PROPN")}
        except Exception:
            pass
    nouns |= set(toks)
    mapped = [LEXICON.get(n, n) for n in nouns]
    mapped = sorted(set([m for m in mapped if 1 <= len(m) <= 30]))
    return mapped[:60]

# ----------------------------
# PART LEXICON (class -> list of part prompts)
# ----------------------------
PART_LEXICON = {
    # people
    "person": ["head","hair","face","hand","arm","leg","foot","torso","shirt","pants","shoe","hat"],
    # animals
    "dog": ["head","ear","eye","nose","mouth","tail","leg","paw","body"],
    "cat": ["head","ear","eye","nose","mouth","tail","leg","paw","whisker","body"],
    "horse": ["head","ear","eye","mane","tail","leg","hoof","body"],
    "cow": ["head","ear","eye","horn","tail","leg","hoof","body"],
    "sheep": ["head","ear","eye","horn","tail","leg","hoof","wool","body"],
    "bird": ["head","eye","beak","wing","tail","leg","claw","body"],
    # vehicles
    "car": ["wheel","tire","window","door","headlight","taillight","mirror","hood","trunk","roof","license plate","bumper"],
    "truck": ["wheel","tire","window","door","headlight","taillight","mirror","bed","bumper"],
    "bus": ["wheel","tire","window","door","headlight","taillight","mirror","roof"],
    "bicycle": ["wheel","tire","handlebar","seat","pedal","chain","frame"],
    "motorbike": ["wheel","tire","handlebar","seat","exhaust","headlight"],
    "train": ["window","door","headlight","carriage","wheel","roof"],
    "boat": ["hull","mast","sail","deck","window","door"],
    # furniture/stuff-like
    "chair": ["backrest","seat","leg","armrest","cushion"],
    "couch": ["backrest","seat","armrest","cushion","leg"],
    "sofa": ["backrest","seat","armrest","cushion","leg"],
    "bed": ["headboard","frame","mattress","pillow","blanket","sheet","leg"],
    "dining table": ["tabletop","leg","drawer"],
    "table": ["tabletop","leg","drawer"],
    "tv": ["screen","bezel","stand"],
    "laptop": ["screen","keyboard","trackpad","hinge","bezel","logo"],
    # misc
    "bottle": ["cap","neck","label","body","base"],
    "cup": ["handle","rim","body","base"],
    "bowl": ["rim","body","base"],
    "backpack": ["strap","pocket","zipper","logo"],
    "umbrella": ["canopy","handle","tip","rib","shaft"],
    "tennis racket": ["head","strings","grip","shaft"],
    "skateboard": ["deck","wheel","truck"],
    "frisbee": ["rim","center"],
}

# default generic parts to try if class not in lexicon
DEFAULT_PARTS = ["part", "component", "region"]

# ----------------------------
# Generic parts (ORB / SLIC)
# ----------------------------
def generic_parts_orb(img_bgr: np.ndarray, box: BBox, max_parts:int=3) -> List[BBox]:
    patch = crop(img_bgr, box)
    if patch.size == 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=800)
    kps = orb.detect(gray, None)
    if not kps or len(kps) < 4:
        # fallback: grid
        H,W = gray.shape
        parts=[]; G = max(1, int(np.sqrt(max_parts)))
        cell_h = max(5, H // G); cell_w = max(5, W // G)
        count=0
        for gy in range(G):
            for gx in range(G):
                if count >= max_parts: break
                x0 = gx*cell_w; y0 = gy*cell_h
                x1 = min(W-1, x0+cell_w); y1 = min(H-1, y0+cell_h)
                parts.append(BBox(box.x0+x0, box.y0+y0, box.x0+x1, box.y0+y1, 1.0, "part"))
                count += 1
            if count >= max_parts: break
        return parts

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    K = min(max_parts, max(1, len(pts)//40))
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(pts)
    parts=[]
    for k in range(K):
        cluster_pts = pts[km.labels_ == k]
        if cluster_pts.shape[0] == 0: continue
        x0 = int(np.clip(cluster_pts[:,0].min()-6, 0, patch.shape[1]-1))
        y0 = int(np.clip(cluster_pts[:,1].min()-6, 0, patch.shape[0]-1))
        x1 = int(np.clip(cluster_pts[:,0].max()+6, 0, patch.shape[1]-1))
        y1 = int(np.clip(cluster_pts[:,1].max()+6, 0, patch.shape[0]-1))
        parts.append(BBox(box.x0+x0, box.y0+y0, box.x0+x1, box.y0+y1, 1.0, "part"))
    return parts

def generic_parts_slic(img_bgr: np.ndarray, box: BBox, max_parts:int=3) -> List[BBox]:
    if not _SKIMAGE_OK:
        return generic_parts_orb(img_bgr, box, max_parts=max_parts)
    patch = crop(img_bgr, box)
    if patch.size == 0:
        return []
    # SLIC superpixels -> merge by KMeans on mean Lab color + centroid
    from skimage.color import rgb2lab
    img_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    H,W = img_rgb.shape[:2]
    n_segs = max(10, min(200, (H*W)//1500))
    seg = slic(img_rgb, n_segments=n_segs, compactness=10, start_label=0)
    # feature per segment
    feats=[]; boxes=[]; ids=np.unique(seg)
    lab = rgb2lab(img_rgb)
    for sid in ids:
        ys, xs = np.where(seg==sid)
        if ys.size < 20: continue
        x0,x1 = int(xs.min()), int(xs.max())
        y0,y1 = int(ys.min()), int(ys.max())
        L = float(np.mean(lab[ys, xs, 0]))
        a = float(np.mean(lab[ys, xs, 1]))
        b = float(np.mean(lab[ys, xs, 2]))
        cx,cy = float(xs.mean()/W), float(ys.mean()/H)
        feats.append([L,a,b,cx,cy])
        boxes.append(BBox(box.x0+x0, box.y0+y0, box.x0+x1, box.y0+y1, 1.0, "part"))
    if not feats:
        return generic_parts_orb(img_bgr, box, max_parts=max_parts)
    feats = np.array(feats, dtype=np.float32)
    K = min(max_parts, max(1, len(boxes)//5))
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(feats)
    sel=[]; out=[]
    for k in range(K):
        idxs = np.where(km.labels_==k)[0]
        if idxs.size == 0: continue
        # choose largest area segment in this cluster
        best=None; bestA=0
        for ii in idxs:
            b = boxes[ii]; A=b.area()
            if A>bestA: bestA=A; best=ii
        if best is not None:
            out.append(boxes[best])
    return out[:max_parts]

# ----------------------------
# Attributes builder
# ----------------------------
def build_attributes(img_bgr: np.ndarray, d: BBox) -> Dict[str, object]:
    patch = crop(img_bgr, d)
    if patch.size == 0:
        return {"dom_color":(0,0,0), "aspect_ratio":0.0, "edge_energy":0.0}
    dom = dominant_color(patch, k=3)
    H,W = patch.shape[:2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    ee = edge_energy(gray)
    return {"dom_color":tuple(map(int,dom)), "aspect_ratio":float(W/max(1,H)), "edge_energy":float(ee)}

# ----------------------------
# Type grouping for manifold
# ----------------------------
def type_group(cls: str) -> str:
    PEOPLE = {"person"}
    ANIMALS = {"dog","cat","horse","sheep","cow","bird","elephant","bear","zebra","giraffe"}
    VEHICLES = {"car","truck","bus","bicycle","motorbike","motorcycle","boat","train","aeroplane","airplane"}
    FURNITURE = {"chair","couch","sofa","bed","dining table","tv","monitor","tvmonitor"}
    FOOD = {"banana","apple","sandwich","orange","broccoli","carrot","pizza","donut","cake"}
    if cls in PEOPLE: return "people"
    if cls in ANIMALS: return "animals"
    if cls in VEHICLES: return "vehicles"
    if cls in FURNITURE: return "furniture"
    if cls in FOOD: return "food"
    if cls in {"yard","park","field","grass","sky","road","building","house","tree","ground"}: return "place"
    return "object"

class FMGraph:
    def __init__(self):
        self.nodes=[]; self.pos={}; self.edges_parent=[]; self.edges_rel=[]; self.match_pairs=[]

    def add_node(self, name, modality, level, typ):
        self.nodes.append({"name":name, "modality":modality, "level":level, "type":typ})
        return len(self.nodes)-1

    def phi_scene_child(self, i:int, n:int, typ_group:str):
        scales = {"people":0.58,"animals":0.56,"vehicles":0.54,"furniture":0.52,"food":0.50,"place":0.50,"object":0.53}
        s = scales.get(typ_group, 0.53)
        R = 1.0; theta = 2*np.pi * (i / max(1,n))
        T = np.array([R*np.cos(theta), 0.7*R*np.sin(theta)]); A = s*np.eye(2)
        return A, T

    def phi_part(self, slot:str):
        if slot.startswith("part"):  return 0.60*np.eye(2), np.array([0.0,-0.5])
        if slot.startswith("attr"):  return 0.60*np.eye(2), np.array([0.0, 0.5])
        # named semantic part
        return 0.60*np.eye(2), np.array([0.4, 0.0])

    def place_child(self, parent, child, A, T):
        self.pos[child] = A @ np.zeros(2) + self.pos[parent] + T
        self.edges_parent.append((parent, child))

    def build(self, img_entities: List[Tuple[str,str]], txt_entities: List[str], img_parts: Dict[int,List[str]], img_attrs: Dict[int,List[str]]):
        sI = self.add_node("scene(I)","I",0,"scene"); self.pos[sI]=np.array([-0.1,0.0])
        sT = self.add_node("scene(T)","T",0,"scene"); self.pos[sT]=np.array([ 0.1,0.0])

        nodes_I=[]; nI=len(img_entities)
        for i,(name,cls) in enumerate(img_entities):
            tg = type_group(cls)
            idx = self.add_node(f"{name}(I)", "I", 1, cls)
            A,T = self.phi_scene_child(i,nI,tg); self.place_child(sI, idx, A, T); nodes_I.append((idx,cls))

        nodes_T=[]; nT=len(txt_entities)
        for i,cls in enumerate(txt_entities):
            tg = type_group(cls)
            idx = self.add_node(f"{cls}(T)", "T", 1, cls)
            A,T = self.phi_scene_child(i,nT,tg); self.place_child(sT, idx, A, T); nodes_T.append((idx,cls))

        for local_i,(node_idx, cls) in enumerate(nodes_I):
            for slot in img_parts.get(local_i, []):
                child = self.add_node(f"{slot}(I)", "I", 2, slot); A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)
            for slot in img_attrs.get(local_i, []):
                child = self.add_node(f"{slot}(I)", "I", 2, slot); A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)

        def chain(nodes):
            buckets={}
            for idx,cls in nodes: buckets.setdefault(cls,[]).append(idx)
            for arr in buckets.values():
                for a,b in zip(arr[:-1], arr[1:]): self.edges_rel.append((a,b,"group"))
        chain(nodes_I); chain(nodes_T)

        self.match_pairs.append((sI,sT))
        dictI={}
        for idx,cls in nodes_I: dictI.setdefault(cls,[]).append(idx)
        dictT={}
        for idx,cls in nodes_T: dictT.setdefault(cls,[]).append(idx)
        for t in set(dictI.keys()).union(dictT.keys()):
            for a,b in zip(dictI.get(t,[]), dictT.get(t,[])): self.match_pairs.append((a,b))

    def plot(self, out_path:str):
        fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111)
        ax.set_title("Fractal Manifold (Entities + Parts for ALL)"); ax.set_aspect('equal','box'); ax.axis('off')
        for (u,v) in self.edges_parent:
            xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle='-', linewidth=1)
        for (u,v,rt) in self.edges_rel:
            xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle=':', linewidth=1)
        for (u,v) in self.match_pairs:
            xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle='--', linewidth=0.8)
        for i,n in enumerate(self.nodes):
            x,y=self.pos[i]; marker='o' if n["modality"]=="I" else '^'
            ms=60 if n["level"]==0 else (50 if n["level"]==1 else 40)
            ax.scatter([x],[y], marker=marker, s=ms); ax.text(x+0.02, y+0.02, n["name"], fontsize=8)
        im=ax.scatter([],[], marker='o', s=60, label='Image'); tx=ax.scatter([],[], marker='^', s=60, label='Text')
        ax.legend(handles=[im,tx], loc='lower right')
        fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)

# ----------------------------
# Panels
# ----------------------------
def build_L0(img_bgr, caption, out_dir):
    fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title("L0 Scene (Image)"); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L0_scene_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    save_text_panel([caption], "L0 Scene (Text)", os.path.join(out_dir,"L0_scene_text.png"))

def build_L1(img_bgr, dets: List[BBox], cap_entities: List[str], out_dir):
    boxes={f"{d.cls}_{i+1}":d for i,d in enumerate(dets)}
    fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
    draw_boxes(ax, img_bgr, boxes, "L1 Entities (Image)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L1_entities_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    lines=["Entities (caption-derived): " + (", ".join(cap_entities) if cap_entities else "(none)")]
    save_text_panel(lines, "L1 Entities (Text)", os.path.join(out_dir,"L1_entities_text.png"))

def build_L2(img_bgr, dets: List[BBox], parts: Dict[int,List[BBox]], attrs: Dict[int,Dict[str,object]], out_dir):
    fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title("L2 Parts & Attributes (Image)"); ax.axis('off')
    for i,d in enumerate(dets):
        x0,y0,x1,y1=d.as_tuple()
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=1))
        at = attrs.get(i, {}); cx,cy = d.center()
        attr_text = f"attr: BGR{at.get('dom_color',(0,0,0))}, ar={at.get('aspect_ratio',0):.2f}, edge={at.get('edge_energy',0):.1f}"
        ax.text(int(cx), int(cy), attr_text, fontsize=7, va='center',
                bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.0), color="white")
        for pj,pb in enumerate(parts.get(i, [])):
            x0,y0,x1,y1=pb.as_tuple()
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
            label = pb.cls if pb.cls else "part"
            ax.text(x0, max(0,y0-3), f"{label}_{pj+1}", fontsize=8, va='bottom',
                    bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=0.8), color="white")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L2_parts_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    save_text_panel(["Parts per entity: semantic (OV) or generic (SLIC/ORB)",
                     "Attributes: dominant color, aspect ratio, edge energy"],
                    "L2 Parts & Attributes (Text)", os.path.join(out_dir,"L2_parts_text.png"))

# ----------------------------
# PART DETECTION orchestrator
# ----------------------------
def detect_semantic_parts_ov(img_path:str, parent_boxes: List[BBox], class_names: List[str], ov_cfg:str, ov_wt:str,
                             box_th:float=0.25, text_th:float=0.25) -> Dict[int,List[BBox]]:
    """
    Use GroundingDINO to predict semantic parts for each entity.
    We build a union prompt list from all classes' PART_LEXICON (dedup) and run once.
    Then assign part boxes to the nearest parent entity whose IoU > 0.1 or center-inside.
    """
    results = defaultdict(list)
    if not _G_DINO_OK:
        return results
    model = load_model(ov_cfg, ov_wt)
    image_source, image = load_image(img_path)
    # Build prompts
    prompts = set()
    for cls in class_names:
        if cls in PART_LEXICON:
            for p in PART_LEXICON[cls]:
                prompts.add(f"{cls} {p}")  # e.g., "car wheel"
                prompts.add(p)             # and raw "wheel" to capture prompts that omit class
        else:
            for p in DEFAULT_PARTS:
                prompts.add(p)
    caption = ". ".join(sorted(prompts)) + "."
    boxes, logits, phrases = predict(model=model, image=image, caption=caption,
                                     box_threshold=box_th, text_threshold=text_th,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
    H,W = image_source.shape[:2]
    ov_parts=[]
    for b,phrase,score in zip(boxes, phrases, logits):
        cx,cy,bw,bh = b
        x0 = int((cx - bw/2) * W); y0 = int((cy - bh/2) * H)
        x1 = int((cx + bw/2) * W); y1 = int((cy + bh/2) * H)
        ov_parts.append(BBox(x0,y0,x1,y1,float(score), phrase))

    # Assign to parents
    for pi, pb in enumerate(parent_boxes):
        for part in ov_parts:
            # center-in-parent or IoU threshold
            cx = int((part.x0+part.x1)//2); cy = int((part.y0+part.y1)//2)
            if pb.contains_point(cx, cy) or pb.iou(part) > 0.10:
                results[pi].append(part)
        # class-aware NMS within each parent
        results[pi] = nms_boxes(results[pi], iou_th=0.5, same_class_only=True)
    return results

def build_generic_parts_for_all(img_bgr: np.ndarray, dets: List[BBox], max_parts:int) -> Dict[int,List[BBox]]:
    out={}
    for i,d in enumerate(dets):
        parts = generic_parts_slic(img_bgr, d, max_parts=max_parts)
        out[i]=parts
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--caption", type=str, required=True)
    ap.add_argument("--out", type=str, default="./fm_out")
    ap.add_argument("--yolo_model", type=str, default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--ov_on", action="store_true", help="Enable GroundingDINO OV detection & part prompts")
    ap.add_argument("--ov_cfg", type=str, default="GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--ov_weights", type=str, default="groundingdino_swint_ogc.pth")
    ap.add_argument("--ov_box_th", type=float, default=0.25)
    ap.add_argument("--ov_text_th", type=float, default=0.25)
    ap.add_argument("--seg_on", action="store_true", help="Enable DeepLabV3 stuff segmentation")
    ap.add_argument("--seg_min_area", type=int, default=800)
    ap.add_argument("--parts", type=int, default=3, help="Max generic parts per entity")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load image
    if not os.path.exists(args.img):
        raise FileNotFoundError(args.img)
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise RuntimeError("Failed to read image.")

    # A) COCO detection
    yolo = YOLODetector(args.yolo_model, args.conf, args.imgsz)
    dets_yolo = yolo.detect(img_bgr)

    # B) OV detection (entities)
    dets_ov = []
    cap_nouns = nouns_from_caption(args.caption)
    if args.ov_on:
        if not _G_DINO_OK:
            print("[WARN] GroundingDINO not installed; skipping OV entities/parts.")
        else:
            try:
                ovdet = OVBDetector(args.ov_cfg, args.ov_weights, args.ov_box_th, args.ov_text_th)
                text_prompt = ". ".join(cap_nouns) + "."
                dets_ov = ovdet.detect(args.img, text_prompt=text_prompt)
            except Exception as e:
                print("[WARN] OV entity detection failed:", e)

    # C) Stuff segmentation
    dets_stuff = []
    if args.seg_on:
        if not _DEEPLAB_OK:
            print("[WARN] torchvision DeepLab not available; skipping segmentation.")
        else:
            try:
                dl = DeepLabStuff()
                dets_stuff = dl.segment(img_bgr, min_area=args.seg_min_area)
            except Exception as e:
                print("[WARN] Segmentation failed:", e)

    # Merge union for L1
    dets = merge_union([dets_yolo, dets_ov, dets_stuff], iou_th=0.6)

    # Quick diagnostics
    print("Detected classes:", Counter([d.cls for d in dets]))
    for d in dets:
        print(f" - {d.cls}: {d.score:.3f} [{d.x0},{d.y0},{d.x1},{d.y1}]")

    # L0
    fig_dir = args.out
    # Caption-derived tokens only for display
    cap_entities = cap_nouns
    def build_L0(img_bgr, caption, out_dir):
        fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title("L0 Scene (Image)"); ax.axis('off')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L0_scene_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
        save_text_panel([caption], "L0 Scene (Text)", os.path.join(out_dir,"L0_scene_text.png"))
    build_L0(img_bgr, args.caption, fig_dir)

    # L1
    def build_L1(img_bgr, dets: List[BBox], cap_entities: List[str], out_dir):
        boxes={f"{d.cls}_{i+1}":d for i,d in enumerate(dets)}
        fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
        draw_boxes(ax, img_bgr, boxes, "L1 Entities (Image)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L1_entities_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
        lines=["Entities (caption-derived): " + (", ".join(cap_entities) if cap_entities else "(none)")]
        save_text_panel(lines, "L1 Entities (Text)", os.path.join(out_dir,"L1_entities_text.png"))
    build_L1(img_bgr, dets, cap_entities, fig_dir)

    # Build attributes for each detection
    attrs_by_det = {i: build_attributes(img_bgr, d) for i,d in enumerate(dets)}

    # Parts per detection
    parts_by_det = build_generic_parts_for_all(img_bgr, dets, max_parts=args.parts)

    # Human extras (optional) -> merge into parts
    if mp_hands or mp_face:
        # hands
        try:
            if mp_hands:
                hands = mp_hands.Hands(static_image_mode=True, max_num_hands=8, min_detection_confidence=0.4)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)
                if res.multi_hand_landmarks:
                    H,W = img_bgr.shape[:2]
                    hand_boxes=[]
                    for lm in res.multi_hand_landmarks:
                        xs=[p.x*W for p in lm.landmark]; ys=[p.y*H for p in lm.landmark]
                        x0,y0 = int(max(0,min(xs)-8)), int(max(0,min(ys)-8))
                        x1,y1 = int(min(W-1,max(xs)+8)), int(min(H-1,max(ys)+8))
                        hand_boxes.append(BBox(x0,y0,x1,y1,1.0,"hand"))
                    # assign to nearest/inside person
                    for hb in hand_boxes:
                        best_i = None; best_iou = 0.0
                        for i,d in enumerate(dets):
                            if d.cls != "person": continue
                            j = d.iou(hb)
                            if j>best_iou: best_iou=j; best_i=i
                        if best_i is not None:
                            parts_by_det.setdefault(best_i, []).append(hb)
        except Exception as e:
            print("[WARN] MediaPipe hands failed:", e)
        # face->hair
        try:
            if mp_face:
                fd = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.35)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                res = fd.process(img_rgb)
                if res.detections:
                    H,W = img_bgr.shape[:2]
                    for det in res.detections:
                        rb = det.location_data.relative_bounding_box
                        x0=int(rb.xmin*W); y0=int(rb.ymin*H); x1=int((rb.xmin+rb.width)*W); y1=int((rb.ymin+rb.height)*H)
                        face = BBox(x0,y0,x1,y1, float(det.score[0]), "face")
                        # hair region above face
                        h = y1-y0; hy1 = max(0, y0 + int(0.15*h)); hy0 = max(0, hy1 - int(0.45*h))
                        hair = BBox(x0, max(0,hy0), x1, max(0,hy1), 1.0, "hair")
                        # assign to nearest person box
                        best_i=None; best=0.0
                        for i,d in enumerate(dets):
                            if d.cls != "person": continue
                            sc = max(d.iou(face), d.iou(hair))
                            if sc>best: best=sc; best_i=i
                        if best_i is not None:
                            parts_by_det.setdefault(best_i, []).extend([face, hair])
        except Exception as e:
            print("[WARN] MediaPipe face/hair failed:", e)

    # Semantic parts (OV) per class (optional)
    if args.ov_on and _G_DINO_OK and os.path.exists(args.ov_cfg) and os.path.exists(args.ov_weights):
        try:
            class_names = [d.cls for d in dets]
            sem_parts = detect_semantic_parts_ov(args.img, dets, class_names, args.ov_cfg, args.ov_weights,
                                                 box_th=args.ov_box_th, text_th=args.ov_text_th)
            # merge: prefer OV parts where present, append generic otherwise
            for i in range(len(dets)):
                ov_list = sem_parts.get(i, [])
                if ov_list:
                    # keep top-N unique labels; ensure not too many
                    ov_list = nms_boxes(ov_list, iou_th=0.5, same_class_only=True)[:max(1, args.parts)]
                parts_by_det[i] = (ov_list + parts_by_det.get(i, []))[:max(1, args.parts)]
        except Exception as e:
            print("[WARN] OV semantic parts failed:", e)

    # L2
    def build_L2(img_bgr, dets: List[BBox], parts: Dict[int,List[BBox]], attrs: Dict[int,Dict[str,object]], out_dir):
        fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title("L2 Parts & Attributes (Image)"); ax.axis('off')
        for i,d in enumerate(dets):
            x0,y0,x1,y1=d.as_tuple()
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=1))
            at = attrs.get(i, {}); cx,cy = d.center()
            attr_text = f"attr: BGR{at.get('dom_color',(0,0,0))}, ar={at.get('aspect_ratio',0):.2f}, edge={at.get('edge_energy',0):.1f}"
            ax.text(int(cx), int(cy), attr_text, fontsize=7, va='center',
                    bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.0), color="white")
            for pj,pb in enumerate(parts.get(i, [])):
                x0,y0,x1,y1=pb.as_tuple()
                ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
                label = pb.cls if pb.cls else "part"
                ax.text(x0, max(0,y0-3), f"{label}_{pj+1}", fontsize=8, va='bottom',
                        bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=0.8), color="white")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L2_parts_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
        save_text_panel(["Parts per entity: semantic (OV where possible) + generic (SLIC/ORB) fallback",
                         "Attributes: dominant color, aspect ratio, edge energy"],
                        "L2 Parts & Attributes (Text)", os.path.join(out_dir,"L2_parts_text.png"))
    build_L2(img_bgr, dets, parts_by_det, attrs_by_det, fig_dir)

    # ----- Fractal manifold (same as before) -----
    class FMGraph:
        def __init__(self):
            self.nodes=[]; self.pos={}; self.edges_parent=[]; self.edges_rel=[]; self.match_pairs=[]
        def add_node(self, name, modality, level, typ):
            self.nodes.append({"name":name, "modality":modality, "level":level, "type":typ}); return len(self.nodes)-1
        def type_group(self, cls: str) -> str:
            return type_group(cls)
        def phi_scene_child(self, i:int, n:int, typ_group:str):
            scales = {"people":0.58,"animals":0.56,"vehicles":0.54,"furniture":0.52,"food":0.50,"place":0.50,"object":0.53}
            s = scales.get(typ_group, 0.53); R = 1.0; theta = 2*np.pi * (i / max(1,n))
            T = np.array([R*np.cos(theta), 0.7*R*np.sin(theta)]); A = s*np.eye(2); return A, T
        def phi_part(self, slot:str):
            if slot.startswith("part"):  return 0.60*np.eye(2), np.array([0.0,-0.5])
            if slot.startswith("attr"):  return 0.60*np.eye(2), np.array([0.0, 0.5])
            return 0.60*np.eye(2), np.array([0.4, 0.0])
        def place_child(self, parent, child, A, T):
            self.pos[child] = A @ np.zeros(2) + self.pos[parent] + T; self.edges_parent.append((parent, child))
        def build(self, img_entities: List[Tuple[str,str]], txt_entities: List[str], img_parts: Dict[int,List[str]], img_attrs: Dict[int,List[str]]):
            sI = self.add_node("scene(I)","I",0,"scene"); self.pos[sI]=np.array([-0.1,0.0])
            sT = self.add_node("scene(T)","T",0,"scene"); self.pos[sT]=np.array([ 0.1,0.0])
            nodes_I=[]; nI=len(img_entities)
            for i,(name,cls) in enumerate(img_entities):
                tg = self.type_group(cls); idx = self.add_node(f"{name}(I)", "I", 1, cls)
                A,T = self.phi_scene_child(i,nI,tg); self.place_child(sI, idx, A, T); nodes_I.append((idx,cls))
            nodes_T=[]; nT=len(txt_entities)
            for i,cls in enumerate(txt_entities):
                tg = self.type_group(cls); idx = self.add_node(f"{cls}(T)", "T", 1, cls)
                A,T = self.phi_scene_child(i,nT,tg); self.place_child(sT, idx, A, T); nodes_T.append((idx,cls))
            for local_i,(node_idx, cls) in enumerate(nodes_I):
                for slot in img_parts.get(local_i, []):
                    child = self.add_node(f"{slot}(I)", "I", 2, slot); A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)
                for slot in img_attrs.get(local_i, []):
                    child = self.add_node(f"{slot}(I)", "I", 2, slot); A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)
            def chain(nodes):
                buckets={}
                for idx,cls in nodes: buckets.setdefault(cls,[]).append(idx)
                for arr in buckets.values():
                    for a,b in zip(arr[:-1], arr[1:]): self.edges_rel.append((a,b,"group"))
            chain(nodes_I); chain(nodes_T)
            self.match_pairs.append((sI,sT))
            dictI={}
            for idx,cls in nodes_I: dictI.setdefault(cls,[]).append(idx)
            dictT={}
            for idx,cls in nodes_T: dictT.setdefault(cls,[]).append(idx)
            for t in set(dictI.keys()).union(dictT.keys()):
                for a,b in zip(dictI.get(t,[]), dictT.get(t,[])): self.match_pairs.append((a,b))
        def plot(self, out_path:str):
            fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111)
            ax.set_title("Fractal Manifold (Entities + Parts for ALL)"); ax.set_aspect('equal','box'); ax.axis('off')
            for (u,v) in self.edges_parent:
                xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle='-', linewidth=1)
            for (u,v,rt) in self.edges_rel:
                xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle=':', linewidth=1)
            for (u,v) in self.match_pairs:
                xu,yu=self.pos[u]; xv,yv=self.pos[v]; ax.plot([xu,xv],[yu,yv], linestyle='--', linewidth=0.8)
            for i,n in enumerate(self.nodes):
                x,y=self.pos[i]; marker='o' if n["modality"]=="I" else '^'
                ms=60 if n["level"]==0 else (50 if n["level"]==1 else 40)
                ax.scatter([x],[y], marker=marker, s=ms); ax.text(x+0.02, y+0.02, n["name"], fontsize=8)
            im=ax.scatter([],[], marker='o', s=60, label='Image'); tx=ax.scatter([],[], marker='^', s=60, label='Text')
            ax.legend(handles=[im,tx], loc='lower right'); fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)

    # Prepare manifold inputs
    img_entities = [(f"{d.cls}_{i+1}", d.cls) for i,d in enumerate(dets)]
    txt_entities = cap_entities
    img_parts_slots = {i: [ (p.cls if p.cls else "part") for p in parts_by_det.get(i, []) ] for i in range(len(dets))}
    img_attrs_slots = {i: ["attr"]*min(2, len(attrs_by_det.get(i, {}))) for i in range(len(dets))}
    fm = FMGraph()
    fm.build(img_entities=img_entities, txt_entities=txt_entities, img_parts=img_parts_slots, img_attrs=img_attrs_slots)
    fm.plot(os.path.join(fig_dir, "fractal_manifold.png"))

    # Dump JSON
    out_json = {
        "detections":[vars(d) for d in dets],
        "caption_nouns": cap_entities,
        "parts_by_det": {str(i): [vars(b) for b in parts_by_det.get(i,[])] for i in range(len(dets))},
        "attrs_by_det": attrs_by_det,
        "backends": {
            "yolo_model": args.yolo_model, "conf": args.conf, "imgsz": args.imgsz,
            "ov_on": args.ov_on, "ov_cfg": args.ov_cfg, "ov_weights": args.ov_weights,
            "seg_on": args.seg_on, "seg_min_area": args.seg_min_area
        }
    }
    with open(os.path.join(fig_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print("[OK] Outputs at:", fig_dir)

if __name__ == "__main__":
    main()
