# -*- coding: utf-8 -*-
"""
Single-image GENERIC Fractal-Manifold builder.
---------------------------------------------
For ANY given image + caption, this script:
1) Detects whatever objects are present (YOLOv8, all classes).
2) L0: shows the scene (image/text).
3) L1: shows detected entities on the image, and caption-mapped entities on the text side.
4) L2: for EACH detected entity, generates generic "parts" (unsupervised via ORB keypoint clustering)
       and generic attributes (dominant color, aspect ratio, edge energy). If 'person' appears and
       MediaPipe is available, it may also add hands/face→hair, but the pipeline remains generic.
5) Builds a type-aware fractal-manifold embedding with shared contractions (per high-level group)
   and plots the graph. Cross-modal matches pair same-type nodes greedily.
6) Saves all figures and a JSON dump of detections/parts/attributes.

Install:
  pip install ultralytics opencv-python matplotlib numpy pillow scikit-learn spacy
  # Optional enhancements:
  pip install mediapipe
  python -m spacy download en_core_web_sm

Run:
  python fm_single_generic.py \
    --img /path/to/image.jpg \
    --caption "Your caption here." \
    --out ./fm_out \
    --yolo_model yolov8l.pt \
    --conf 0.30 \
    --parts 3
"""
import os, json, argparse, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

# Optional modules
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

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

def nms(boxes: List[BBox], iou_th=0.5) -> List[BBox]:
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b:b.score, reverse=True)
    kept=[]
    while boxes:
        b = boxes.pop(0); kept.append(b)
        boxes = [x for x in boxes if b.iou(x) < iou_th]
    return kept

def draw_boxes(ax, img_bgr: np.ndarray, boxes: Dict[str,BBox], title:str):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    for name,b in boxes.items():
        x0,y0,x1,y1 = b.as_tuple()
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
        ax.text(x0, max(0,y0-4), name, fontsize=9, va='bottom')
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
# Detection
# ----------------------------
class GenericDetector:
    def __init__(self, model_path:str, conf:float):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf = conf
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=8, min_detection_confidence=0.4) if mp_hands else None
        self.face  = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.35) if mp_face else None

    def detect_all(self, img_bgr: np.ndarray) -> List[BBox]:
        res = self.model.predict(img_bgr[..., ::-1], verbose=False, conf=self.conf)[0]  # RGB
        out=[]
        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x0,y0,x1,y1 = map(int, b); cls=res.names[int(c)]
            out.append(BBox(x0,y0,x1,y1,float(s),cls))
        return nms(out, 0.5)

    def detect_hands(self, img_bgr: np.ndarray) -> List[BBox]:
        if not self.hands: return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb); boxes=[]
        if res.multi_hand_landmarks:
            H,W = img_bgr.shape[:2]
            for lm in res.multi_hand_landmarks:
                xs=[p.x*W for p in lm.landmark]; ys=[p.y*H for p in lm.landmark]
                x0,y0 = int(max(0,min(xs)-8)), int(max(0,min(ys)-8))
                x1,y1 = int(min(W-1,max(xs)+8)), int(min(H-1,max(ys)+8))
                boxes.append(BBox(x0,y0,x1,y1,1.0,"hand"))
        return boxes

    def detect_faces(self, img_bgr: np.ndarray) -> List[BBox]:
        if not self.face: return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face.process(img_rgb); boxes=[]
        if res.detections:
            H,W = img_bgr.shape[:2]
            for det in res.detections:
                rb = det.location_data.relative_bounding_box
                x0=int(rb.xmin*W); y0=int(rb.ymin*H)
                x1=int((rb.xmin+rb.width)*W); y1=int((rb.ymin+rb.height)*H)
                boxes.append(BBox(x0,y0,x1,y1,float(det.score[0]),"face"))
        return boxes

def derive_hair_from_face(face_box: BBox, img_h:int) -> BBox:
    x0,y0,x1,y1 = face_box.as_tuple(); h = y1-y0
    hair_h = int(0.45*h); hy1 = max(0, y0 + int(0.15*h)); hy0 = max(0, hy1-hair_h)
    hx0,hx1 = x0, x1
    return BBox(hx0, max(0,hy0), hx1, min(img_h-1,hy1), 1.0, "hair")

# ----------------------------
# Generic parts via ORB clustering
# ----------------------------
def generic_parts_from_orb(img_bgr: np.ndarray, box: BBox, max_parts:int=3) -> List[BBox]:
    patch = crop(img_bgr, box)
    if patch.size == 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    kps = orb.detect(gray, None)
    if not kps or len(kps) < 4:
        # fallback: simple grid-based pseudo-parts
        H,W = gray.shape
        parts=[]
        G = max(1, int(np.sqrt(max_parts)))
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
    K = min(max_parts, max(1, len(pts)//30))
    if K <= 0: K = 1
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

# ----------------------------
# Caption parsing (generic mapping to classes)
# ----------------------------
LEXICON = {
    "man":"person", "men":"person", "woman":"person", "women":"person",
    "guy":"person", "boy":"person", "girl":"person", "people":"person", "person":"person",
    "dog":"dog", "cat":"cat", "horse":"horse", "cow":"cow", "sheep":"sheep", "bird":"bird",
    "car":"car", "truck":"truck", "bus":"bus", "bicycle":"bicycle", "motorcycle":"motorbike", "motorbike":"motorbike", "train":"train", "boat":"boat", "airplane":"aeroplane", "plane":"aeroplane",
    "bench":"bench", "chair":"chair", "sofa":"couch", "couch":"couch", "bed":"bed", "table":"dining table", "tv":"tv", "monitor":"tv", "laptop":"laptop",
    "keyboard":"keyboard", "mouse":"mouse", "cell":"cell phone", "phone":"cell phone", "remote":"remote",
    "bottle":"bottle", "cup":"cup", "bowl":"bowl", "wine":"wine glass", "glass":"wine glass",
    "backpack":"backpack", "handbag":"handbag", "umbrella":"umbrella", "tie":"tie",
    "frisbee":"frisbee","skis":"skis","snowboard":"snowboard","kite":"kite",
    "baseball":"baseball bat","bat":"baseball bat","glove":"baseball glove","tennis":"tennis racket","racket":"tennis racket",
    "book":"book","clock":"clock","vase":"vase","scissors":"scissors","toilet":"toilet",
    # generic mentions
    "hair":"hair","hand":"hand","hands":"hand",
    "yard":"yard","grass":"yard","garden":"yard","field":"yard","park":"yard",
}

def parse_caption(caption:str) -> Dict[str,List[str]]:
    out = {"entities": [], "attributes": [], "relations": []}
    c = caption.strip()
    toks = re.findall(r"[A-Za-z]+", c.lower())
    # try spaCy adjectives/nouns
    if _SPACY_OK:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(caption)
            adjs = [t.lemma_.lower() for t in doc if t.pos_=="ADJ"]
            out["attributes"].extend(adjs)
        except Exception:
            pass
    mapped = []
    for t in toks:
        if t in LEXICON:
            mapped.append(LEXICON[t])
    # uniq
    out["entities"] = sorted(list(set(mapped)))
    if "left" in toks or "right" in toks:
        out["relations"].append("left/right")
    if "look" in toks and ("hand" in toks or "hands" in toks):
        out["relations"].append("look_at(person, hand)")
    return out

# ----------------------------
# Type grouping for contractions
# ----------------------------
PEOPLE = {"person"}
ANIMALS = {"dog","cat","horse","sheep","cow","bird","elephant","bear","zebra","giraffe"}
VEHICLES = {"car","truck","bus","bicycle","motorbike","motorcycle","boat","train","aeroplane","airplane"}
FURNITURE = {"chair","couch","sofa","bed","dining table","tv","monitor","tvmonitor"}
FOOD = {"banana","apple","sandwich","orange","broccoli","carrot","pizza","donut","cake"}

def type_group(cls: str) -> str:
    if cls in PEOPLE: return "people"
    if cls in ANIMALS: return "animals"
    if cls in VEHICLES: return "vehicles"
    if cls in FURNITURE: return "furniture"
    if cls in FOOD: return "food"
    if cls in {"yard","park","field","grass"}: return "place"
    return "object"

# ----------------------------
# Fractal manifold
# ----------------------------
class FMGraph:
    def __init__(self):
        self.nodes=[]; self.pos={}; self.edges_parent=[]; self.edges_rel=[]; self.match_pairs=[]

    def add_node(self, name, modality, level, typ):
        self.nodes.append({"name":name, "modality":modality, "level":level, "type":typ})
        return len(self.nodes)-1

    def phi_scene_child(self, i:int, n:int, typ_group:str):
        # position children on a ring; contraction (scale) depends on group
        scales = {"people":0.58,"animals":0.56,"vehicles":0.54,"furniture":0.52,"food":0.50,"place":0.50,"object":0.53}
        s = scales.get(typ_group, 0.53)
        R = 1.0
        theta = 2*np.pi * (i / max(1,n))
        T = np.array([R*np.cos(theta), 0.7*R*np.sin(theta)])
        A = s*np.eye(2)
        return A, T

    def phi_part(self, slot:str):
        # same across types for generic parts/attrs
        if slot.startswith("part"):  # ORB cluster part
            return 0.60*np.eye(2), np.array([0.0, -0.5])
        if slot.startswith("attr"):  # attribute node
            return 0.60*np.eye(2), np.array([0.0,  0.5])
        # default
        return 0.60*np.eye(2), np.array([0.0, 0.0])

    def place_child(self, parent, child, A, T):
        self.pos[child] = A @ np.zeros(2) + self.pos[parent] + T
        self.edges_parent.append((parent, child))

    def build(self, img_entities: List[Tuple[str,str]], txt_entities: List[str], img_parts: Dict[int,List[str]], img_attrs: Dict[int,List[str]]):
        # roots
        sI = self.add_node("scene(I)","I",0,"scene"); self.pos[sI]=np.array([-0.1,0.0])
        sT = self.add_node("scene(T)","T",0,"scene"); self.pos[sT]=np.array([ 0.1,0.0])

        # L1 image
        nodes_I=[]  # (idx, cls)
        nI=len(img_entities)
        for i,(name,cls) in enumerate(img_entities):
            tg = type_group(cls)
            idx = self.add_node(f"{name}(I)", "I", 1, cls)
            A,T = self.phi_scene_child(i,nI,tg); self.place_child(sI, idx, A, T)
            nodes_I.append((idx, cls))

        # L1 text
        nodes_T=[]
        nT=len(txt_entities)
        for i,cls in enumerate(txt_entities):
            tg = type_group(cls)
            idx = self.add_node(f"{cls}(T)", "T", 1, cls)
            A,T = self.phi_scene_child(i,nT,tg); self.place_child(sT, idx, A, T)
            nodes_T.append((idx, cls))

        # L2 image parts/attrs
        for local_i,(node_idx, cls) in enumerate(nodes_I):
            part_list = img_parts.get(local_i, [])
            for j,slot in enumerate(part_list):
                child = self.add_node(f"{slot}(I)", "I", 2, slot)
                A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)
            attr_list = img_attrs.get(local_i, [])
            for j,slot in enumerate(attr_list):
                child = self.add_node(f"{slot}(I)", "I", 2, slot)
                A,T = self.phi_part(slot); self.place_child(node_idx, child, A, T)

        # relations: chain within same class on each side
        def chain_by_class(nodes):
            bucket={}
            for idx,cls in nodes:
                bucket.setdefault(cls, []).append(idx)
            for _,arr in bucket.items():
                for a,b in zip(arr[:-1], arr[1:]):
                    self.edges_rel.append((a,b,"group"))
        chain_by_class(nodes_I); chain_by_class(nodes_T)

        # cross-modal matches by class (greedy by order)
        self.match_pairs.append((sI,sT))
        dictI={}
        for idx,cls in nodes_I: dictI.setdefault(cls,[]).append(idx)
        dictT={}
        for idx,cls in nodes_T: dictT.setdefault(cls,[]).append(idx)
        all_types=set(dictI.keys()).union(dictT.keys())
        for t in all_types:
            li, lt = dictI.get(t,[]), dictT.get(t,[])
            for a,b in zip(li, lt):
                self.match_pairs.append((a,b))

    def centroid_distance_by_level(self, level:int) -> float:
        I = [self.pos[i] for i,n in enumerate(self.nodes) if n["level"]==level and n["modality"]=="I"]
        T = [self.pos[i] for i,n in enumerate(self.nodes) if n["level"]==level and n["modality"]=="T"]
        if not I or not T: return float("nan")
        cI=np.mean(np.stack(I),axis=0); cT=np.mean(np.stack(T),axis=0)
        return float(np.linalg.norm(cI-cT))

    def plot(self, out_path:str):
        fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111)
        ax.set_title("Fractal Manifold (type-aware, generic)"); ax.set_aspect('equal','box'); ax.axis('off')
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

def build_L1(img_bgr, dets: List[BBox], cap_info: Dict[str,List[str]], out_dir):
    boxes={f"{d.cls}_{i+1}":d for i,d in enumerate(dets)}
    fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
    draw_boxes(ax, img_bgr, boxes, "L1 Entities (Image)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L1_entities_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    lines=[
        "Entities (from caption mapping): " + (", ".join(cap_info["entities"]) if cap_info["entities"] else "(none)"),
        "Attributes: " + (", ".join(cap_info["attributes"]) if cap_info["attributes"] else "(none)"),
        "Relations: " + (", ".join(cap_info["relations"]) if cap_info["relations"] else "(none)"),
    ]
    save_text_panel(lines, "L1 Entities (Text)", os.path.join(out_dir,"L1_entities_text.png"))

def build_L2(img_bgr, dets: List[BBox], parts: Dict[int,List[BBox]], attrs: Dict[int,Dict[str,object]], extra_parts: List[BBox], out_dir):
    fig=plt.figure(figsize=(4,6)); ax=fig.add_subplot(111)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title("L2 Parts & Attributes (Image)"); ax.axis('off')
    # draw entity boxes lightly
    for i,d in enumerate(dets):
        x0,y0,x1,y1=d.as_tuple()
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=1))
        # attributes text near center
        at = attrs.get(i, {})
        cx,cy = d.center()
        attr_text = f"attr: BGR{at.get('dom_color',(0,0,0))}, ar={at.get('aspect_ratio',0):.2f}, edge={at.get('edge_energy',0):.1f}"
        ax.text(int(cx), int(cy), attr_text, fontsize=7, va='center')
    # draw parts
    pcount=0
    for i,plist in parts.items():
        for pj,pb in enumerate(plist):
            x0,y0,x1,y1=pb.as_tuple()
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
            ax.text(x0, max(0,y0-3), f"part_{i}_{pj+1}", fontsize=8, va='bottom')
            pcount+=1
    # extra parts (e.g., hands/hair) if any
    for j,b in enumerate(extra_parts):
        x0,y0,x1,y1=b.as_tuple()
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
        ax.text(x0, max(0,y0-3), f"{b.cls}_{j+1}", fontsize=8, va='bottom')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"L2_parts_image.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    save_text_panel(["Generic parts: ORB-clustered subregions per entity",
                     "Generic attributes: dominant color, aspect ratio, edge energy (per entity)",
                     "Optional extras: hands / hair if visible (not mandatory)"],
                    "L2 Parts & Attributes (Text)", os.path.join(out_dir,"L2_parts_text.png"))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--caption", type=str, required=True)
    ap.add_argument("--out", type=str, default="./fm_out")
    ap.add_argument("--yolo_model", type=str, default="yolov8l.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--parts", type=int, default=3, help="Max generic parts per entity via ORB clustering")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load
    if not os.path.exists(args.img):
        raise FileNotFoundError(args.img)
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise RuntimeError("Failed to read image.")

    # detectors
    det = GenericDetector(args.yolo_model, args.conf)

    # detect everything
    dets  = det.detect_all(img_bgr)
    hands = det.detect_hands(img_bgr)  # optional, only if visible
    faces = det.detect_faces(img_bgr)  # optional, only if visible
    hair_boxes = [derive_hair_from_face(f, img_bgr.shape[0]) for f in faces] if faces else []

    # L0
    build_L0(img_bgr, args.caption, args.out)

    # caption parse
    cap_info = parse_caption(args.caption)

    # L1
    build_L1(img_bgr, dets, cap_info, args.out)

    # build generic parts/attrs for each detection
    parts_by_det: Dict[int,List[BBox]] = {}
    attrs_by_det: Dict[int,Dict[str,object]] = {}
    for i,d in enumerate(dets):
        # parts
        plist = generic_parts_from_orb(img_bgr, d, max_parts=args.parts)
        # assign any visible hands to this entity if they lie inside
        for hb in hands:
            # check overlap
            if d.iou(hb) > 0.05:
                plist.append(hb)
        # if face/hair boxes overlap and class is person, include
        if d.cls == "person":
            for hb in hair_boxes:
                if d.iou(hb) > 0.05:
                    plist.append(hb)
        parts_by_det[i] = plist

        # attributes
        patch = crop(img_bgr, d)
        if patch.size == 0:
            attrs_by_det[i] = {"dom_color":(0,0,0), "aspect_ratio":0.0, "edge_energy":0.0}
        else:
            dom = dominant_color(patch, k=3)
            H,W = patch.shape[:2]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            ee = edge_energy(gray)
            attrs_by_det[i] = {"dom_color":dom, "aspect_ratio":float(W/max(1,H)), "edge_energy":ee}

    # L2
    # collect extras for display that aren't already counted (e.g., global hands/hair not overlapping any det)
    extras = []
    for hb in hands:
        if max((hb.iou(d) for d in dets), default=0.0) < 0.05:
            extras.append(hb)
    for hb in hair_boxes:
        if max((hb.iou(d) for d in dets), default=0.0) < 0.05:
            extras.append(hb)

    build_L2(img_bgr, dets, parts_by_det, attrs_by_det, extras, args.out)

    # Fractal manifold
    img_entities = [(f"{d.cls}_{i+1}", d.cls) for i,d in enumerate(dets)]
    txt_entities = cap_info["entities"]
    # map parts/attrs into slot names for FM graph (just labels)
    img_parts_slots = {}
    img_attrs_slots = {}
    for i,(name,cls) in enumerate(img_entities):
        img_parts_slots[i] = []
        for j,pb in enumerate(parts_by_det.get(i, [])):
            slot = "part" if pb.cls in {"part","hand","hair"} else "part"
            img_parts_slots[i].append(f"{slot}")
        img_attrs_slots[i] = [f"attr" for _ in range(min(2, len(attrs_by_det.get(i,{}))))]

    fm = FMGraph()
    fm.build(img_entities=img_entities, txt_entities=txt_entities, img_parts=img_parts_slots, img_attrs=img_attrs_slots)
    fm.plot(os.path.join(args.out, "fractal_manifold.png"))

    # Dump JSON
    out_json = {
        "detections":[vars(d) for d in dets],
        "hands":[vars(h) for h in hands],
        "faces":[vars(f) for f in faces],
        "hair_boxes":[vars(h) for h in hair_boxes],
        "parts_by_det": {str(i): [vars(b) for b in parts_by_det.get(i,[])] for i in range(len(dets))},
        "attrs_by_det": attrs_by_det,
        "caption_info": cap_info
    }
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print("[OK] Outputs at:", args.out)

if __name__ == "__main__":
    main()
