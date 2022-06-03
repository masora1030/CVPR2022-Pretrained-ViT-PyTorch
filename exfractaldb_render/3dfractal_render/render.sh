#! /bin/bash
variance_threshold=0.05
numof_category=10
param_path='./3Dfractal/3DIFS_cat1000'
save_path='./3Dfractal/cat1000_ins145'

# Parameter search
python code/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python code/instance.py --load_root ${param_path} --save_root ${save_path} --classes ${numof_category}