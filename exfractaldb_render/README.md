# 3D Fractal DataBase (3D-FractalDB) 

## Summary

The repository contains a 3D Fractal Category Search, Multi-View Fractal DataBase (MV-FractalDB) and Point Cloud Fractal DataBase (PC-FractalDB) Construction in Python3.

The repository is based on the paper:<br>
Ryosuke Yamada, Ryo Takahashi, Ryota Suzuki, Akio Nakamura, Yusuke Yoshiyasu, Ryusuke Sagawa and Hirokatsu Kataoka, <br>
"MV-FractalDB: Formula-driven Supervised Learning for Multi-view Image Recognition" <br>
International Conference on Intelligent Robots and Systems (IROS) 2021 <br>
[[Project](https://ryosuke-yamada.github.io/Multi-view-Fractal-DataBase/)] 
[[PDF](https://ieeexplore.ieee.org/abstract/document/9635946)]<br>

Ryosuke Yamada, Hirokatsu Kataoka, Naoya Chiba, Yukiyasu Domae and Tetsuya Ogata<br>
"Point Cloud Pre-training with Natural 3D Structure"<br>
International Conference on Computer Vision and Pattern Recognition (CVPR) 2022 <br>
[[Project](https://ryosuke-yamada.github.io/PointCloud-FractalDataBase/)] 
[[PDF]()]<br>

<!-- Run the python script ```render.sh```, you can get 3D fractal models and multi-view fractal images. -->

<!-- ## Prerequisites
- Anaconda
- Python 3.9+ -->

## Installation
1. Create conda virtual environment.
```
$ conda create -n mvfdb python=3.9 -y
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

## PC-FractalDB Construction
Coming Soon...

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

## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST) and Tokyo Denki University (TDU) are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the datas and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.

## License
MIT