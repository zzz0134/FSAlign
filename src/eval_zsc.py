import os, json, torch
from .config import get_zsc_args
from .utils.common import load_embeddings
from .models.heads import ProjectionHead
from .models.fractal_kernel import FractalKernel

def main():
    args = get_zsc_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img_feats, txt_feats, img_ids, txt_ids, splits = load_embeddings(args.emb_dir)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    d_img = img_feats.size(1)
    d_txt = txt_feats.size(1)
    proj_dim = ckpt['args']['proj_dim']

    img_head = ProjectionHead(d_img, out_dim=proj_dim).to(device)
    txt_head = ProjectionHead(d_txt, out_dim=proj_dim).to(device)
    img_head.load_state_dict(ckpt['img_head']); img_head.eval()
    txt_head.load_state_dict(ckpt['txt_head']); txt_head.eval()

    fk = FractalKernel(num_scales=ckpt['args']['num_scales'],
                       r_min=ckpt['args']['r_min'], r_max=ckpt['args']['r_max'],
                       alpha_mode=ckpt['args']['alpha_mode'], alpha_fixed=ckpt['args']['alpha_fixed'],
                       learn_Q=False, Q_init=ckpt['fk'].get('Q', 3.0), device=device).to(device)
    fk.load_state_dict(ckpt['fk'], strict=False); fk.eval()

    # prompts_file: mapping class -> indices into txt_embeddings (demo). Replace with your prompt encoding for real ZSC.
    with open(args.prompts_file,'r') as f:
        prompt_dict = json.load(f)  # {class_name: [indices ...]}

    class2proto = {}
    for cls, idxs in prompt_dict.items():
        idxs = [int(i) for i in idxs]
        class2proto[cls] = txt_feats[idxs].to(device)  # (M_c, d_txt)

    Zi = img_head(img_feats.to(device))  # (N_img, proj_dim)
    preds = []
    with torch.no_grad():
        for i in range(Zi.size(0)):
            z = Zi[i:i+1]
            best_cls, best_E = None, 1e9
            for cls, proto_txt in class2proto.items():
                proto = txt_head(proto_txt)  # project to shared space
                E = fk.dfrac_point_distribution(z, proto).item()
                if E < best_E:
                    best_E = E
                    best_cls = cls
            preds.append(best_cls)

    print('Zero-shot classification predictions (first 10):', preds[:10])

if __name__ == '__main__':
    main()
