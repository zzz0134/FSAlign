# demo_fractal_hierarchy_alignment_viz.py
# Author: modality-gap project
# Hierarchy (both modalities): Global -> Entities -> Relations
# - Image Global: CLIP(image)
# - Image Entities: Faster R-CNN detections -> crop -> CLIP(crop)
# - Image Relations: pairwise spatial relations; embedding = z_i + z_j + Proj(geom)
# - Text Global: CLIP(caption)
# - Text Entities: spaCy noun chunks (fallback regex)
# - Text Relations: (head, relation, tail) via dependency (fallback heuristic)
# Alignment: initial entity matches (Hungarian on cosine), add global anchor, then Orthogonal Procrustes (text->image)
# Visualization: BEFORE/AFTER scatter (PCA) for Entities & Relations; image entity boxes; relation overlay; gap report.

import os, math, re, json, argparse, itertools, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as TVF

import open_clip
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# ----------------- Utils -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def l2norm_np(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_sim(A, B):
    A = l2norm_np(A, -1)
    B = l2norm_np(B, -1)
    return A @ B.T

def draw_boxes(img: Image.Image, boxes: List[Tuple[int,int,int,int]], labels: List[str], color=(0,255,0), width=3):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for (x0,y0,x1,y1), lab in zip(boxes, labels):
        d.rectangle([x0,y0,x1,y1], outline=color, width=width)
        if lab:
            d.text((x0+2, y0+2), lab, fill=color)
    return im

def draw_relations(img: Image.Image, boxes: List[Tuple[int,int,int,int]], rels: List[Tuple[int,int,str]]):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for (i,j,rel) in rels:
        (x0,y0,x1,y1) = boxes[i]
        (u0,v0,u1,v1) = boxes[j]
        xi, yi = (x0+x1)//2, (y0+y1)//2
        xj, yj = (u0+u1)//2, (v0+v1)//2
        d.line([xi, yi, xj, yj], width=2, fill=(255,0,0))
        xm, ym = (xi+xj)//2, (yi+yj)//2
        d.text((xm, ym), rel, fill=(255,0,0))
    return im


# ----------------- CLIP -----------------
def build_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

def encode_image_patches(model, preprocess, device, patches: List[Image.Image], batch_size=64):
    if len(patches) == 0:
        return np.zeros((0,512), dtype=np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            ims = torch.stack([preprocess(p) for p in patches[i:i+batch_size]]).to(device)
            f = model.encode_image(ims)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)

def encode_texts(model, tokenizer, device, texts: List[str], batch_size=128):
    if not texts:
        return np.zeros((0,512), dtype=np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            toks = tokenizer(texts[i:i+batch_size]).to(device)
            f = model.encode_text(toks)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ----------------- Detection -----------------
COCO_CATEGORIES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@dataclass
class DetConfig:
    score_thresh: float = 0.6
    top_k: int = 6
    nms_iou: float = 0.5

def run_detector(img: Image.Image, cfg: DetConfig, device="cuda"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()
    with torch.no_grad():
        x = TVF.to_tensor(img).to(device)
        out = model([x])[0]
    boxes = out["boxes"].detach().cpu()
    scores = out["scores"].detach().cpu()
    labels = out["labels"].detach().cpu()

    # Filter by score
    keep = scores >= cfg.score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # NMS
    keep_idx = nms(boxes, scores, cfg.nms_iou)
    boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    # Top-k
    if len(scores) > cfg.top_k:
        topk_idx = torch.topk(scores, cfg.top_k).indices
        boxes, scores, labels = boxes[topk_idx], scores[topk_idx], labels[topk_idx]

    # Format
    boxes = boxes.round().int().tolist()
    names = [COCO_CATEGORIES[l] if l < len(COCO_CATEGORIES) else f"id{int(l)}" for l in labels.tolist()]
    return boxes, names, scores.tolist()


# ----------------- Geometry + Relations -----------------
def geom_features(boxA, boxB):
    # 6-dim geometry feature:
    # [dx, dy, log(wA/wB), log(hA/hB), IoU, dist_norm]
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    aw, ah = ax1-ax0, ay1-ay0
    bw, bh = bx1-bx0, by1-by0
    acx, acy = (ax0+ax1)/2.0, (ay0+ay1)/2.0
    bcx, bcy = (bx0+bx1)/2.0, (by0+by1)/2.0
    dx, dy = (acx-bcx), (acy-bcy)
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    areaA = aw*ah
    areaB = bw*bh
    union = max(1.0, areaA + areaB - inter)
    iou = inter / union
    dist = math.sqrt(dx*dx + dy*dy) / (math.sqrt(aw*aw+ah*ah) + math.sqrt(bw*bw+bh*bh) + 1e-6)
    return np.array([dx, dy, math.log((aw+1e-6)/(bw+1e-6)), math.log((ah+1e-6)/(bh+1e-6)), iou, dist], dtype=np.float32)

def spatial_relation_label(boxA, boxB):
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    acx, acy = (ax0+ax1)//2, (ay0+ay1)//2
    bcx, bcy = (bx0+bx1)//2, (by0+by1)//2
    dx, dy = bcx - acx, bcy - acy
    horiz = "right of" if dx>0 else "left of"
    vert = "below" if dy>0 else "above"
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    aw, ah = ax1-ax0, ay1-ay0
    bw, bh = bx1-bx0, by1-by0
    union = max(1.0, aw*ah + bw*bh - inter)
    iou = inter/union
    if iou > 0.05:
        return "overlapping"
    dist = math.sqrt(dx*dx+dy*dy)
    diagA = math.sqrt(aw*aw+ah*ah)
    diagB = math.sqrt(bw*bw+bh*bh)
    if dist < 0.5*(diagA+diagB):
        return f"near & {horiz}/{vert}"
    else:
        return f"{horiz}/{vert}"


# ----------------- Text NLP -----------------
def extract_text_entities_relations(caption: str):
    """
    Prefer spaCy: noun chunks as entities; (subj, rel, obj) as relations.
    Fallback: heuristic splits.
    """
    caption = caption.strip()
    text_entities = []
    text_relations = []
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.load("en_core_web_md")
        doc = nlp(caption)

        ents = []
        for nc in doc.noun_chunks:
            s = nc.text.strip()
            if len(s)>=2:
                ents.append(s)
        seen=set()
        for e in ents:
            el = e.lower()
            if el not in seen:
                text_entities.append(e)
                seen.add(el)

        for tok in doc:
            if tok.dep_ in ("ROOT","relcl") and tok.pos_ in ("VERB","AUX"):
                subj = [w for w in tok.lefts if w.dep_ in ("nsubj","nsubjpass")]
                obj  = [w for w in tok.rights if w.dep_ in ("dobj","attr","oprd","pobj","dative")]
                if subj:
                    head = subj[0].subtree
                    head_text = " ".join([w.text for w in head])
                else:
                    head_text = ""
                rel_tokens = [tok.lemma_]
                rel_tokens += [w.text for w in tok.children if w.dep_ in ("prep","prt")]
                rel_text = " ".join(rel_tokens)
                tail_text = ""
                if obj:
                    tail = obj[0].subtree
                    tail_text = " ".join([w.text for w in tail])
                if head_text and tail_text and rel_text:
                    text_relations.append((head_text, rel_text, tail_text))

        if not text_relations:
            m = re.split(r"\b(with|in|on|at|to|from|near|beside|between)\b", caption, flags=re.IGNORECASE)
            if len(m) >= 3:
                a = m[0].strip(); prep = m[1]; b = m[2].strip()
                if a and b:
                    text_relations.append((a, prep, b))

    except Exception:
        chunks = re.split(r",| and ", caption)
        for ch in chunks:
            ch = ch.strip()
            if 1 <= len(ch.split()) <= 4:
                text_entities.append(ch)
        m = re.search(r"([A-Za-z ]+?)\s+(look|hold|play|watch|touch|talk|stand|sit|walk|hang)\w*\s+(.*)", caption, flags=re.I)
        if m:
            text_relations.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))

    text_entities = text_entities[:8]
    text_relations = text_relations[:10]
    return text_entities, text_relations


# ----------------- Alignment -----------------
def orthogonal_procrustes(X, Y):
    # Map X->Y with orthogonal R and translation t
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    t = Y.mean(axis=0) - X.mean(axis=0) @ R
    return R, t


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--out_dir", default="outputs_hier_demo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--det_score", type=float, default=0.6)
    ap.add_argument("--det_topk", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"

    # 1) Load inputs
    img = Image.open(args.image).convert("RGB")
    caption = args.caption.strip()

    # 2) Build CLIP
    model, preprocess, tokenizer = build_clip(args.clip_model, args.clip_pretrained, device)

    # 3) Global embeddings
    with torch.no_grad():
        img_global_t = model.encode_image(preprocess(img).unsqueeze(0).to(device))
        img_global_t = img_global_t / (img_global_t.norm(dim=-1, keepdim=True) + 1e-8)
        img_global = img_global_t.float().cpu().numpy()[0]

        txt_global_t = model.encode_text(tokenizer([caption]).to(device))
        txt_global_t = txt_global_t / (txt_global_t.norm(dim=-1, keepdim=True) + 1e-8)
        txt_global = txt_global_t.float().cpu().numpy()[0]

    # 4) Image entities (detection)
    det_cfg = DetConfig(score_thresh=args.det_score, top_k=args.det_topk)
    boxes, names, scores = run_detector(img, det_cfg, device=device)
    vis_ent = draw_boxes(img, boxes, names, color=(0,255,0), width=3)
    vis_ent.save(os.path.join(args.out_dir, "image_entities.png"))

    # Entity crops & embeddings
    crops = [img.crop(b) for b in boxes]
    feats_i_entities = encode_image_patches(model, preprocess, device, crops)

    # 5) Text entities & relations
    text_entities, text_relations = extract_text_entities_relations(caption)
    feats_t_entities = encode_texts(model, tokenizer, device, text_entities)

    # 6) Image relations (pairwise spatial)
    rel_triplets_img = []  # (i,j, relation_label)
    geom_feats = []
    for i, j in itertools.combinations(range(len(boxes)), 2):
        rel = spatial_relation_label(boxes[i], boxes[j])
        rel_triplets_img.append((i,j, rel))
        geom_feats.append(geom_features(boxes[i], boxes[j]))

    # Compose image pair embeddings: z_i + z_j + Proj(geom)
    d = feats_i_entities.shape[1] if feats_i_entities.size else 512
    rng = np.random.default_rng(123)
    Wgeom = rng.standard_normal((6, d)).astype(np.float32) * 0.05
    feats_i_relations = []
    for (i,j,_), g in zip(rel_triplets_img, geom_feats):
        vec = feats_i_entities[i] + feats_i_entities[j] + g @ Wgeom
        vec = l2norm_np(vec[None,:])[0]
        feats_i_relations.append(vec)
    feats_i_relations = np.array(feats_i_relations, dtype=np.float32) if rel_triplets_img else np.zeros((0,d), dtype=np.float32)

    vis_rel = draw_relations(img, boxes, rel_triplets_img[:20])
    vis_rel.save(os.path.join(args.out_dir, "image_relations.png"))

    # 7) Text relations -> phrases "head REL tail"
    rel_phrases_text = [f"{h} {r} {t}" for (h,r,t) in text_relations]
    feats_t_relations = encode_texts(model, tokenizer, device, rel_phrases_text)

    # 8) Initial entity matching (Hungarian on cosine sim)
    if len(feats_t_entities)>0 and len(feats_i_entities)>0:
        S_ent = cosine_sim(feats_t_entities, feats_i_entities)  # (n_text_ent, n_img_ent)
        cost = -S_ent
        ri, cj = linear_sum_assignment(cost)
        anchors_t = []
        anchors_i = []
        for r,c in zip(ri, cj):
            if S_ent[r,c] > 0:  # conservative
                anchors_t.append(feats_t_entities[r])
                anchors_i.append(feats_i_entities[c])
        anchors_t = np.stack(anchors_t) if anchors_t else np.zeros((0,d), dtype=np.float32)
        anchors_i = np.stack(anchors_i) if anchors_i else np.zeros((0,d), dtype=np.float32)
        if anchors_t.shape[0] == 0:
            anchors_t = txt_global[None,:]
            anchors_i = img_global[None,:]
        else:
            anchors_t = np.vstack([anchors_t, txt_global[None,:]])
            anchors_i = np.vstack([anchors_i, img_global[None,:]])
        R, t = orthogonal_procrustes(anchors_t, anchors_i)
    else:
        R, t = orthogonal_procrustes(txt_global[None,:], img_global[None,:])

    def map_text(Zt):
        if Zt.size == 0:
            return Zt
        return l2norm_np(Zt @ R + t, -1)

    # 9) BEFORE/AFTER for visualization & gap
    Zi_ent = feats_i_entities
    Zt_ent_b = feats_t_entities
    Zt_ent_a = map_text(feats_t_entities)

    Zi_rel = feats_i_relations
    Zt_rel_b = feats_t_relations
    Zt_rel_a = map_text(feats_t_relations)

    # 10) PCA to 2D (fit jointly per level for fair comparison)
    def pca_2d_fit_transform(A_list):
        X = np.vstack([A for A in A_list if A.size>0])
        if X.shape[0] < 2:
            return [np.zeros((A.shape[0],2)) for A in A_list]
        pca = PCA(n_components=2).fit(X)
        return [pca.transform(A) if A.size>0 else np.zeros((0,2)) for A in A_list]

    Xi_ent_2d, Xt_ent_b_2d, Xt_ent_a_2d = pca_2d_fit_transform([Zi_ent, Zt_ent_b, Zt_ent_a])
    Xi_rel_2d, Xt_rel_b_2d, Xt_rel_a_2d = pca_2d_fit_transform([Zi_rel, Zt_rel_b, Zt_rel_a])

    # 11) Plots
    def scatter_two(Xa, Xb, lab_a, lab_b, title, path, marker_a="o", marker_b="x"):
        plt.figure(figsize=(6,6))
        if Xa.shape[0]>0:
            plt.scatter(Xa[:,0], Xa[:,1], s=30, alpha=0.85, label=lab_a, marker=marker_a)
        if Xb.shape[0]>0:
            plt.scatter(Xb[:,0], Xb[:,1], s=30, alpha=0.85, label=lab_b, marker=marker_b)
        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(path, dpi=150); plt.close()

    scatter_two(Xi_ent_2d, Xt_ent_b_2d, "Image entities", "Text entities (before)", "Entities: BEFORE alignment", os.path.join(args.out_dir, "entities_before.png"))
    scatter_two(Xi_ent_2d, Xt_ent_a_2d, "Image entities", "Text entities (after)",  "Entities: AFTER alignment (shared manifold)", os.path.join(args.out_dir, "entities_after.png"))
    scatter_two(Xi_rel_2d, Xt_rel_b_2d, "Image relations", "Text relations (before)", "Relations: BEFORE alignment", os.path.join(args.out_dir, "relations_before.png"))
    scatter_two(Xi_rel_2d, Xt_rel_a_2d, "Image relations", "Text relations (after)",  "Relations: AFTER alignment (shared manifold)", os.path.join(args.out_dir, "relations_after.png"))

    # 12) Numeric gap (centroid distances) for each level
    def centroid(X):
        return X.mean(axis=0, keepdims=True) if X.size>0 else np.zeros((1, X.shape[1] if X.ndim==2 else 2))

    def gap_dict(Zi, Ztb, Zta, name):
        Xi2, XtB2, XtA2 = pca_2d_fit_transform([Zi, Ztb, Zta])
        cb = float(np.linalg.norm(centroid(Xi2) - centroid(XtB2)))
        ca = float(np.linalg.norm(centroid(Xi2) - centroid(XtA2)))
        return name, {"before": cb, "after": ca, "improved": (cb - ca)}

    gaps = {}
    gaps["global"] = {
        "before": float(1.0 - float((img_global @ txt_global).sum())),  # 1 - cosine, proxy
        "after":  float(1.0 - float((img_global @ l2norm_np((txt_global @ R + t)[None,:])[0]).sum()))
    }
    for (name, Zi, Ztb, Zta) in [
        ("entities", Zi_ent, Zt_ent_b, Zt_ent_a),
        ("relations", Zi_rel, Zt_rel_b, Zt_rel_a),
    ]:
        k, v = gap_dict(Zi, Ztb, Zta, name)
        gaps[k] = v

    with open(os.path.join(args.out_dir, "gap_report.json"), "w") as f:
        json.dump(gaps, f, indent=2)

    # 13) Save artifacts
    with open(os.path.join(args.out_dir, "text_entities.json"), "w") as f:
        json.dump(text_entities, f, indent=2)
    with open(os.path.join(args.out_dir, "text_relations.json"), "w") as f:
        json.dump(text_relations, f, indent=2)
    with open(os.path.join(args.out_dir, "image_entities.json"), "w") as f:
        json.dump([{"box": b, "label": n, "score": s} for b,n,s in zip(boxes, names, scores)], f, indent=2)

    with open(os.path.join(args.out_dir, "README_levels.txt"), "w") as f:
        f.write(
            "Hierarchy used in this demo:\n"
            "- Global: whole image ↔ whole caption.\n"
            "- Entities: detected objects (image) ↔ noun chunks (text).\n"
            "- Relations: spatial relations between object pairs (image) ↔ subject–relation–object phrases (text).\n\n"
            "Alignment: initial entity correspondences via Hungarian on cosine similarity (text↔image),\n"
            "plus a global anchor; then Orthogonal Procrustes maps text embeddings into the image space.\n"
            "BEFORE/AFTER plots visualize gap reduction per level.\n"
        )

    print("[OK] Outputs saved to:", args.out_dir)
    print("Files:")
    for name in ["image_entities.png","image_relations.png","entities_before.png","entities_after.png",
                 "relations_before.png","relations_after.png","gap_report.json",
                 "text_entities.json","text_relations.json","image_entities.json","README_levels.txt"]:
        print(" -", name)


if __name__ == "__main__":
    main()
