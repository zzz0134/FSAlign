import os, json, torch

def load_embeddings(emb_dir):
    img = torch.load(os.path.join(emb_dir, 'img_embeddings.pt'), map_location='cpu')
    txt = torch.load(os.path.join(emb_dir, 'txt_embeddings.pt'), map_location='cpu')
    with open(os.path.join(emb_dir, 'img_ids.json'),'r') as f:
        img_ids = json.load(f)
    with open(os.path.join(emb_dir, 'txt_ids.json'),'r') as f:
        txt_ids = json.load(f)
    with open(os.path.join(emb_dir, 'split_indices.json'),'r') as f:
        splits = json.load(f)
    return img, txt, img_ids, txt_ids, splits
