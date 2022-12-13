# Extended Fractal DataBase (ExFractalDB)
Run the bash script ```ExFractalDB_render.sh```, you can get our Multi-view Fractal DataBase.

[MV-FractalDB](https://ryosuke-yamada.github.io/Multi-view-Fractal-DataBase/) generates 12 images from _fixed_ viewpoints,
whereas ExFractalDB _randomly_ selects and projects 2D images from 3D models.
Note that this code is for constructing the MV-FractalDB, whereas you can construct the ExFractalDB by changing the view-point at random.

## Requirements

* Python 3.x (worked at 3.7)

* openexr-python (worked at 1.3.2)

* PySDL2 (worked at 0.9.10)

* PyOpenGL (worked at 3.1.5)

* PyOpenGL-accelerate (worked at 3.1.5)

* plyfile (worked at 0.7.4)

* PIL (worked at 9.0.0)

## Running the code

We prepared execution file ExFractalDB_render.sh in this directory. 
The execution file contains our recommended parameters. 
Please type the following commands on your environment. 
You can execute the fractal category search, the 3D fractal model generate, and the multi-view image render, ExFractalDB Construction.

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

> **Note**
> 
> - We use OpenGL to generate ExFractalDB, and because of this, generating ExFractalDB in remote environment is currently not tested. Please try to generate ExFractalDB in your local environment.
