# ExFractalDB 

## Installation
1. Create anaconda virtual environment.
```
$ conda create -n exfdb python=3.x -y
$ conda activate exfdb
```

2. Install requirement modules
```
$ conda install -c conda-forge openexr-python
$ pip install -r requirements.txt
```

## Running the code

We prepared execution file ExFractalDB_render.sh in the top directory. 
The execution file contains our recommended parameters. 
Please type the following commands on your environment. 
You can execute the fractal category search, the 3D fractal model generate, and the multi-view image render, MV-FractalDB Construction.

```bash ExFractalDB_render.sh```

The folder structure is constructed as follows.

```misc
./
  ExFractalDB/
    3DIFS_param/
        ExFractalDB-{category}/
            000000.csv
            000001.csv
            ...
    3Dmodel/
        ExFractalDB-{category}/
           000000/
           000000_0000.ply
           000000_0001.ply
           ...
         ...
    images/
        ExFractalDB-{category}/
           000000/
           000000_00000_000.png
           000000_00001_001.png
           ...
         ...
```