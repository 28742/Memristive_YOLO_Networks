#!/bin/bash

# Define the ADC resolution bits to test
# 4 to 16 bits with step 2 -> 4, 6, 8, 10, 12, 14, 16
bits=("4" "6" "8" "10" "12" "14" "16")

echo "Starting parameter sweep for radc-bits: ${bits[*]}"

for b in "${bits[@]}"
do
    echo "--------------------------------------------------"
    echo "Running inference with radc-bits=$b"
    echo "--------------------------------------------------"
    
    # Log file for this run
    logfile="inference_radc_${b}bit.log"
    
    # Run python script
    # Note: We use --var 0 to isolate the effect of ADC resolution (optional, remove --var 0 to use default 0.05)
    # The user didn't specify var, so I will honestly just leave it to default (0.05) or maybe 0 is better?
    # Usually you want to see the effect of quantization alone.
    # But let's just stick to varying radc-bits mostly.
    # To be safe and consistent with "ideal" memristor except for ADC, I'd set var to 0.
    # But if they want realistic simulation, 0.05 is default.
    # Let's run with default var (0.05) effectively unless specified.
    # Actually, previous request was sweeping var.
    # I will just run with --mem-enabled and --radc-bits.
    
    python /home/zrc/MemIntelli/examples/11_yolov5_coco_inference.py --mem-enabled --radc-bits "$b" > "$logfile" 2>&1
    
    echo "Completed radc-bits=$b. Output saved to $logfile"
done

echo "All tests finished."
