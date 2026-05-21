import os, json, argparse, torch
from PIL import Image
from tqdm import tqdm
import open_clip
from torchvision import transforms
from .data.flickr30k import parse_captions, load_karpathy_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--images_dir', type=str, default='flickr30k-images')
    ap.add_argument('--captions_file', type=str, default='results_20130124.token')
    ap.add_argument('--split_json', type=str, default='karpathy_splits.json')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--clip_model', type=str, default='ViT-B-32')
    ap.add_argument('--clip_pretrained', type=str, default='openai')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    img_dir = os.path.join(args.data_root, args.images_dir)
    cap_path = os.path.join(args.data_root, args.captions_file)
    split_path = os.path.join(args.data_root, args.split_json)

    captions = parse_captions(cap_path)  # dict fn -> list[str]
    splits = load_karpathy_split(split_path)

    image_fns = sorted([fn for fn in os.listdir(img_dir) if fn.lower().endswith(('.jpg','.jpeg','.png'))])
    fn2idx = {fn:i for i,fn in enumerate(image_fns)}

    img_ids = image_fns
    txt_entries = []
    for fn, caps in captions.items():
        if fn not in fn2idx:
            continue
        for ci, c in enumerate(caps):
            txt_entries.append((fn, ci, c))

    # Encode images
    img_feats = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(image_fns), bs), desc='Images'):
            batch_fns = image_fns[i:i+bs]
            batch_imgs = []
            for fn in batch_fns:
                im = Image.open(os.path.join(img_dir, fn)).convert('RGB')
                batch_imgs.append(preprocess(im))
            batch = torch.stack(batch_imgs).to(args.device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            img_feats.append(feats.float().cpu())
    img_feats = torch.cat(img_feats, dim=0)

    # Encode texts
    txt_feats = []
    txt_ids = []
    with torch.no_grad():
        for i in tqdm(range(0, len(txt_entries), bs), desc='Texts'):
            batch = txt_entries[i:i+bs]
            texts = [b[2] for b in batch]
            tokens = tokenizer(texts).to(args.device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            txt_feats.append(feats.float().cpu())
            for (fn,ci,_) in batch:
                txt_ids.append({'image_fn': fn, 'cap_idx': int(ci)})
    txt_feats = torch.cat(txt_feats, dim=0)

    torch.save(img_feats, os.path.join(args.out_dir, 'img_embeddings.pt'))
    torch.save(txt_feats, os.path.join(args.out_dir, 'txt_embeddings.pt'))
    with open(os.path.join(args.out_dir, 'img_ids.json'),'w') as f:
        json.dump(img_ids, f)
    with open(os.path.join(args.out_dir, 'txt_ids.json'),'w') as f:
        json.dump(txt_ids, f)

    with open(os.path.join(args.out_dir, 'split_indices.json'),'w') as f:
        json.dump(splits, f)

if __name__ == '__main__':
    main()
