#!/bin/bash

# Define the variance values to test
# 0 to 0.2 with step 0.04 -> 0, 0.04, 0.08, 0.12, 0.16, 0.2
vars=( "0.06" "0.08")

echo "Starting parameter sweep for var: ${vars[*]}"

for v in "${vars[@]}"
do
    echo "--------------------------------------------------"
    echo "Running inference with var=$v"
    echo "--------------------------------------------------"
    
    # Log file for this run
    logfile="inference_var_${v}.log"
    
    # Execute the python script with the current var
    # Adding --mem-enabled as it seems required to use DPETensor where var is used
    # Assuming the user wants to run in the current environment
    python /home/zrc/MemIntelli/examples/11_yolov5_coco_inference.py --mem-enabled --var "$v" > "$logfile" 2>&1
    
    echo "Completed var=$v. Output saved to $logfile"
done

echo "All tests finished."
