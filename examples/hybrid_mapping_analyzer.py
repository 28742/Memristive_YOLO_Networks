import copy
import torch
import torch.nn as nn
from memintelli.NN_layers import Conv2dMem

class DiffConv2dMem(nn.Module):
    """
    Differential Pair architecture wrapper for a Conv2dMem layer.
    Replaces negative weights with positive values mapped to a negative array.
    """
    def __init__(self, c_mem):
        super().__init__()
        self.pos_conv = copy.deepcopy(c_mem)
        self.neg_conv = copy.deepcopy(c_mem)
        w = c_mem.weight.data
        self.pos_conv.weight = nn.Parameter(torch.clamp(w, min=0))
        self.neg_conv.weight = nn.Parameter(torch.clamp(-w, min=0))
        
        # Bias is only needed once, zero it out on the negative conv
        if self.neg_conv.bias is not None:
            self.neg_conv.bias = nn.Parameter(torch.zeros_like(self.neg_conv.bias.data))
            
    def forward(self, x):
        return self.pos_conv(x) - self.neg_conv(x)
        
    def update_weight(self):
        if hasattr(self.pos_conv, 'update_weight'):
            self.pos_conv.update_weight()
        if hasattr(self.neg_conv, 'update_weight'):
            self.neg_conv.update_weight()


def analyze_and_apply_hybrid_mapping(model, tau_thresh=0.01, density_threshold=0.2, zero_thresh_fallback=0.01):
    """
    Analyzes the density of small negative weights [-tau_thresh, 0) per layer.
    If density exceeds density_threshold, the layer is mapped to a Diff-Pair (2x array)
    architecture to avoid 2's complement conductance collapse.
    Otherwise, it is kept as a Single-Array (offset or cropped) to save 50% device area.
    
    Args:
        model: YOLOv5 PyTorch model instance (already containing Conv2dMem).
        tau_thresh: The boundary defining a "small negative number" prone to complement errors.
        density_threshold: The threshold density. Layer density > this -> Diff Pair mode.
        zero_thresh_fallback: For single-array layers, any small negative up to this value is zeroed out
                              to suppress localized noise spikes.
    """
    print(f"\n[{'MixMap Analyzer'}] Starting Layer-wise Density Analysis")
    print(f"[{'MixMap Analyzer'}] Parameters: Tau={tau_thresh}, Density Threshold={density_threshold}, Single-Array Prune Threshold={zero_thresh_fallback}")
    
    total_area_full_diff = 0
    total_area_hybrid = 0
    
    diff_layer_count = 0
    single_layer_count = 0
    
    # We will replace modules in-place, so we write a recursive helper
    def recursive_analyze_replace(module, prefix=""):
        nonlocal total_area_full_diff, total_area_hybrid
        nonlocal diff_layer_count, single_layer_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Conv2dMem):
                weight = child.weight.data
                total_els = weight.numel()
                
                # Analyze small negative density
                small_neg_mask = (weight < 0) & (weight >= -tau_thresh)
                small_neg_count = small_neg_mask.sum().item()
                density = small_neg_count / (total_els + 1e-9)
                
                # Base area proxy is the number of weights
                # A full diff pair architecture would double this unconditionally
                base_area = total_els
                total_area_full_diff += (base_area * 2)
                
                if density >= density_threshold:
                    # High risk of noise: Use Mode A (Differential Pair)
                    total_area_hybrid += (base_area * 2)
                    diff_layer_count += 1
                    
                    # Apply diff pair wrapper
                    diff_wrapper = DiffConv2dMem(child)
                    setattr(module, name, diff_wrapper)
                    print(f"    [Mode A: Diff] {full_name:30s} | Density: {density:.3f} >= {density_threshold:.3f}")
                else:
                    # Low risk: Use Mode B (Single Array)
                    total_area_hybrid += (base_area * 1)
                    single_layer_count += 1
                    
                    # Apply optional fallback zeroing to prevent the remaining few negatives from causing spikes
                    if zero_thresh_fallback > 0.0:
                        zero_mask = (weight < 0) & (weight >= -zero_thresh_fallback)
                        weight[zero_mask] = 0.0
                        
                    print(f"    [Mode B: Sngl] {full_name:30s} | Density: {density:.3f} <  {density_threshold:.3f}")
                    
            else:
                recursive_analyze_replace(child, full_name)
                
    recursive_analyze_replace(model)
    
    # Calculate savings
    area_savings_ratio = 1.0 - (total_area_hybrid / float(total_area_full_diff))
    
    print("\n[{'MixMap Analyzer'}] ------------ Analysis Sub-Report ------------")
    print(f"[{'MixMap Analyzer'}] Mode A (Diff-Pair) Layers Assigned  : {diff_layer_count}")
    print(f"[{'MixMap Analyzer'}] Mode B (Single-Array) Layers Assigned : {single_layer_count}")
    print(f"[{'MixMap Analyzer'}] Total Equivalent Array Area Saved     : = {area_savings_ratio * 100:.2f}% (compared to pure Diff-Pair)")
    print("[{'MixMap Analyzer'}] -----------------------------------------------\n")
    
    return model
