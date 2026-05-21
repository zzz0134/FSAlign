import torch, math

class FractalKernel(torch.nn.Module):
    """Fractal diffusion kernel over multiple scales with spectral weighting.

    K(z,z') = sum_k w_k * exp(-||z - z'||^2 / (4 r_k)),  w_k ∝ r_k^{-alpha},
    alpha = 1 + Q/2 (Q: spectral dimension, learnable).
    """
    def __init__(self, num_scales=8, r_min=0.1, r_max=5.0, alpha_mode='spectral', alpha_fixed=1.5, learn_Q=True, Q_init=3.0, device='cpu'):
        super().__init__()
        self.num_scales = num_scales
        self.register_buffer('r', torch.exp(torch.linspace(math.log(r_min), math.log(r_max), num_scales)))
        self.alpha_mode = alpha_mode
        if learn_Q:
            self.Q = torch.nn.Parameter(torch.tensor(float(Q_init)))
        else:
            self.register_buffer('Q_buf', torch.tensor(float(Q_init)))
            self.Q = None
        self.alpha_fixed = alpha_fixed
        self.device = device

    def alpha(self):
        if self.alpha_mode == 'fixed':
            return torch.tensor(self.alpha_fixed, device=self.r.device)
        else:
            Q = self.Q if self.Q is not None else self.Q_buf
            return 1.0 + 0.5 * torch.clamp(Q, min=0.5, max=10.0)

    def weights(self):
        a = self.alpha()
        w = self.r.pow(-a)
        w = w / (w.sum() + 1e-8)
        return w

    def kernel(self, Z, Zp):
        sq = (Z[:,None,:] - Zp[None,:,:]).pow(2).sum(-1)
        K = 0.0
        w = self.weights()
        for k in range(self.num_scales):
            rk = self.r[k]
            K = K + w[k] * torch.exp(- sq / (4.0 * rk + 1e-8))
        return K

    def dfrac_point_point(self, z, zp):
        Kxy = self.kernel(z, zp)
        wsum = self.weights().sum()
        const = 2.0 * wsum
        D = const - 2.0 * Kxy
        return D

    def dfrac_point_distribution(self, z, P):
        KzP = self.kernel(z, P)
        KPP = self.kernel(P, P)
        wsum = self.weights().sum()
        term1 = wsum * torch.ones(z.size(0), device=z.device)
        term2 = 2.0 * KzP.mean(dim=1)
        term3 = KPP.mean() * torch.ones(z.size(0), device=z.device)
        D = term1 - term2 + term3
        return D
