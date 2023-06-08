#! /bin/bash
export PYOPENGL_PLATFORM=egl
variance_threshold=0.05
numof_category=1000
param_path='../dataset/MVFractalDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/MVFractalDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/MVFractalDB-'${numof_category}'/images'

# Parameter search
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}

# Render Multi-view images
python image_render/render.py --load_root ${model_save_path} --save_root ${image_save_path}
