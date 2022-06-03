#! /bin/bash
variance_threshold=0.05
numof_category=1000
param_path='./../datasets/MVFractalDB/3DIFS_params/MVFractalDB-'${numof_category}
3dmodel_save_path='./../datasets/MVFractalDB/3D-model/MVFractalDB-'${numof_category}
img_save_path='./../datasets/MVFractalDB/images/MVFractalDB-'${numof_category}

# Parameter search
python category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python instance.py --load_root ${param_path} --save_root ${3dmodel_save_path} --classes ${numof_category}

# Multi-view images render
python render.py --load_root ${3dmodel_save_path} --save_root ${img_save_path}

# MV-FractalDB Pre-training
# python pretraining/main.py --path2traindb=${save_path} --dataset='FractalDB-'${numof_category} --numof_classes=${numof_category} --usenet=${arch}

# MV-FractalDB Fine-tuning
# python finetuning/main.py --path2db=${save_path} --path2weight='./data/weight' --dataset='FractalDB-'${numof_category} --ft_dataset='CIFAR10' --numof_pretrained_classes=${numof_category} --usenet=${arch}
