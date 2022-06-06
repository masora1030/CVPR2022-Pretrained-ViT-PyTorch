#! /bin/bash
variance_threshold=0.05
numof_category=1000
param_path='./../MVFractalDB/3DIFS_params/MVFractalDB-'${numof_category}
3dmodel_save_path='./../MVFractalDB/3Dmodels/MVFractalDB-'${numof_category}
image_save_path='./../MVFractalDB/images/MVFractalDB-'${numof_category}

# Parameter search
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${3dmodel_save_path} --classes ${numof_category}

# Render Multi-view images
python image_render/render.py --load_root ${3dmodel_save_path} --save_root ${image_save_path}