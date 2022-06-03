# Multi-view Fractal Database (MV-FractalDB) and Extended Fractal Database (ExFractalDB) 

## Summary
The repository contains a 3D Fractal Category Search, Multi-View Fractal DataBase (MV-FractalDB) and Extended Fractal DataBase (ExFractalDB) Construction in Python3.

## Installation
1. Create conda virtual environment.
```
$ conda create -n mvfdb python=3.x -y
$ conda activate mvfdb
```

2. Install requirement modules
```
$ conda install -c conda-forge openexr-python
$ pip install -r requirements.txt
```

## MV-FractalDB Construction
1. Search fractal category and create a 3D fractal model
```
$ cd 3dfractal_render
$ bash render.sh
```

2. Render multi-view image
```
$ cd image_render
$ python render.py
```

## Citation
If you use this code, please cite the following paper:

```bibtex
@inproceedings{yamada2021mv,
  title={MV-FractalDB: Formula-driven Supervised Learning for Multi-view Image Recognition},
  author={Yamada, Ryosuke and Takahashi, Ryo and Suzuki, Ryota and Nakamura, Akio and Yoshiyasu, Yusuke and Sagawa, Ryusuke and Kataoka, Hirokatsu},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2076--2083},
  organization={IEEE}
}
```