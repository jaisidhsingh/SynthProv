#!/bin/bash

# Set the range of the loop
start=1
end=3000

# Loop from start to end
for ((counter=start; counter<=end; counter++)); do
	python3 chunk.py $counter;
done
