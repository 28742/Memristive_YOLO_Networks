import torch
import sys
sys.path.insert(0, '/home/zrc/MemIntelli')
from memintelli.NN_layers import Conv2dMem
from memintelli.pimpy.memmat_tensor import DPETensor

engine = DPETensor(device='cpu', write_variation=0.05)
conv = Conv2dMem(engine, 64, 64, 3, input_slice=(1,1,2,4), weight_slice=(1,1,2,4), device='cpu')

W = conv.weight.data
conv.update_weight()
import numpy as np

# normal snr
# SlicedData.quantized_data is pseudo-FP weight
quant_w = conv.weight_sliced.quantized_data.detach()
G_noisy = conv.weight_sliced.G.detach()
sw = conv.weight_sliced.sliced_weights.detach().view(1, 1, -1, 1, 1)
max_data = conv.weight_sliced.max_data.detach()
max_weights = conv.weight_sliced.sliced_max_weights.detach().view(1, 1, -1, 1, 1)
G_ideal = (G_noisy - engine.LGS) / engine.Q_G
G_norm = G_ideal / (engine.g_level - 1)
val_reconstructed = torch.sum((G_norm * max_weights) * sw, dim=2)
total_range = torch.tensor(2**(8 - 1) - 1, dtype=torch.float32)
noise_w_recovered = (val_reconstructed / total_range) * max_data.view(-1, 1, 1, 1)
noise_w_recovered = noise_w_recovered.transpose(1, 2).reshape(W.shape[0]*9, W.shape[1])[:W.shape[0], :].flatten()
orig_w_flat = W.view(W.shape[0], -1).t()[:W.shape[1], :].flatten()

mse = torch.mean((orig_w_flat - noise_w_recovered)**2)
print("Normal Two's Complement SNR:", 10 * torch.log10(torch.sum(orig_w_flat**2) / mse).item())

# Diff Pair
W_pos = torch.clamp(W, min=0)
W_neg = torch.clamp(-W, min=0)

import copy
pos_conv = copy.deepcopy(conv)
neg_conv = copy.deepcopy(conv)
pos_conv.weight.data = W_pos
neg_conv.weight.data = W_neg

pos_conv.update_weight()
neg_conv.update_weight()

def get_rec(c):
    qw = c.weight_sliced.quantized_data.detach()
    Gn = c.weight_sliced.G.detach()
    sw = c.weight_sliced.sliced_weights.detach().view(1, 1, -1, 1, 1)
    md = c.weight_sliced.max_data.detach()
    mw = c.weight_sliced.sliced_max_weights.detach().view(1, 1, -1, 1, 1)
    Gi = (Gn - engine.LGS) / engine.Q_G
    G_no = Gi / (engine.g_level - 1)
    vr = torch.sum((G_no * mw) * sw, dim=2)
    nr = (vr / total_range) * md.view(-1, 1, 1, 1)
    nr = nr.transpose(1, 2).reshape(W.shape[0]*9, W.shape[1])[:W.shape[0], :].flatten()
    return nr

diff_rec = get_rec(pos_conv) - get_rec(neg_conv)
diff_mse = torch.mean((orig_w_flat - diff_rec)**2)
print("Differential Pair SNR:", 10 * torch.log10(torch.sum(orig_w_flat**2) / diff_mse).item())

