
import yaml

sensitive_layers = {3, 4, 9, 10, 12, 16, 57, 58}
total_layers = 60

layers_config = []

for i in range(total_layers):
    entry = {"index": i}
    
    if i in sensitive_layers:
        # Sensitive Configuration: Robust against Write Variation
        # Use single-bit cells (1-bit slicing) to avoid multi-level state variation issues.
        # This uses more cells but is more reliable.
        entry["weight_slice"] = [1, 1, 1, 1, 1, 1, 1, 1]
        entry["input_slice"] = [1, 1, 1, 1, 1, 1, 1, 1]
        # entry["adc"] = 14 # Suggesting higher ADC precision (if supported)
        
        # Maybe varying quantization granularity
        entry["weight_quant_gran"] = [32, 1] 
        entry["input_quant_gran"] = [1, 32]
        
    else:
        # Standard Configuration: Efficient
        # Use multi-bit cells (4-bit, 2-bit) to save area/energy.
        entry["weight_slice"] = [1, 1, 2, 4] 
        entry["input_slice"] = [1, 1, 2, 4]
        
        # Standard granularity
        entry["weight_quant_gran"] = [64, 1]
        entry["input_quant_gran"] = [1, 64]

    layers_config.append(entry)

config = {"layers": layers_config}

with open("/home/zrc/MemIntelli/examples/replacement_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=None, sort_keys=False)

print("Generated replacement_config.yaml")
