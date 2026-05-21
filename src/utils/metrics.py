import numpy as np

def recall_at_k(sim, gt, Ks=(1,5,10)):
    N = sim.shape[0]
    ranks = []
    for i in range(N):
        order = np.argsort(-sim[i])
        g = gt[i]
        rank = min([np.where(order==idx)[0][0] for idx in g])
        ranks.append(rank)
    recalls = {}
    for K in Ks:
        recalls[f'R@{K}'] = float(np.mean([r < K for r in ranks]))
    recalls['MedR'] = float(np.median([r+1 for r in ranks]))
    recalls['MeanR'] = float(np.mean([r+1 for r in ranks]))
    return recalls
