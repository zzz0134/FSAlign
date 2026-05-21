import os, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .config import get_train_args
from .utils.common import load_embeddings
from .models.heads import ProjectionHead
from .models.fractal_kernel import FractalKernel

class PairDataset(Dataset):
    def __init__(self, img_feats, txt_feats, img_ids, txt_ids, splits):
        fn2imgidx = {fn:i for i,fn in enumerate(img_ids)}
        self.pos_pairs = []
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        img_train = set(splits.get('train',{}).get('images', list(range(img_feats.size(0)))))
        txt_train = set(splits.get('train',{}).get('texts',  list(range(txt_feats.size(0)))))
        for j, meta in enumerate(txt_ids):
            fn = meta['image_fn']
            if fn in fn2imgidx:
                i = fn2imgidx[fn]
                if i in img_train and j in txt_train:
                    self.pos_pairs.append((i,j))

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        i, j = self.pos_pairs[idx]
        return i, j

def info_nce(scores, temperature=0.07):
    logits_i2t = scores / temperature
    logits_t2i = scores.T / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_i2t, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_t2i, labels)
    return 0.5 * (loss_i2t + loss_t2i)

def main():
    args = get_train_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img_feats, txt_feats, img_ids, txt_ids, splits = load_embeddings(args.emb_dir)

    ds = PairDataset(img_feats, txt_feats, img_ids, txt_ids, splits)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    d_img = img_feats.size(1)
    d_txt = txt_feats.size(1)
    img_head = ProjectionHead(d_img, out_dim=args.proj_dim).to(device)
    txt_head = ProjectionHead(d_txt, out_dim=args.proj_dim).to(device)

    fk = FractalKernel(num_scales=args.num_scales, r_min=args.r_min, r_max=args.r_max,
                       alpha_mode=args.alpha_mode, alpha_fixed=args.alpha_fixed,
                       learn_Q=args.learn_Q, Q_init=args.Q_init, device=device).to(device)

    optim = torch.optim.AdamW([*img_head.parameters(), *txt_head.parameters(), *fk.parameters()],
                              lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best = 1e9

    for epoch in range(1, args.epochs+1):
        img_head.train(); txt_head.train(); fk.train()
        pbar = tqdm(dl, desc=f'Epoch {epoch}')
        running = 0.0
        for step, (i_idx, j_idx) in enumerate(pbar):
            i_emb = img_feats[i_idx].to(device)
            j_emb = txt_feats[j_idx].to(device)

            zi = img_head(i_emb)
            zj = txt_head(j_emb)

            D = fk.dfrac_point_point(zi, zj)  # (B,B)
            S = -D

            loss = info_nce(S, temperature=0.07)
            optim.zero_grad(); loss.backward(); optim.step()

            running += float(loss.item())
            if (step+1) % args.log_every == 0:
                pbar.set_postfix({'loss': running/args.log_every})
                running = 0.0

        ckpt = {'epoch': epoch, 'img_head': img_head.state_dict(), 'txt_head': txt_head.state_dict(),
                'fk': fk.state_dict(), 'args': vars(args)}
        torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch}.pt'))
        if loss.item() < best:
            best = loss.item()
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))

if __name__ == '__main__':
    main()
