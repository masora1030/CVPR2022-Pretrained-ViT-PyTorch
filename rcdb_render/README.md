# Radial Contour DataBase (RCDB)
Run the python script ```make_rcdb.py```, you can get our Radial Contour dataset.

## Requirements

* noise (worked at 1.2.2)

* PIL (worked at 9.0.0)

* Python 3 (worked at 3.7)


## Running the code

Basically, you can run the code with the command.

```bash
python make_RCDB.py
```

The folder structure is constructed as follows.

```misc
./
  rcdb/
    image/
      00000/
        00000_0000.png
        00000_0001.png
        ...
      00001/
        00001_0000.png
        00001_0001.png
        ...
      ...
    ...
```

You can change the dataset folder name with ```--save_root```. For a faster execution, you shuold run the bash as follows. You must adjust the thread parameter ```--numof_thread``` in the script depending on your computational resource.

```bash
SAVE_ROOT="../dataset/rcdb"
CLASSES=1000
INSTANCES=1000
VERTEX_NUM=200
PERLIN_MIN=0
LINE_WIDTH=0.1
RADIUS_MIN=0
OVAL_RATE=2
START_POS=400
NUMOF_THREAD=10

# Multi-thread processing
for ((i=0 ; i<${NUMOF_THREAD} ; i++))
do
    python make_rcdb.py --save_root=${SAVE_ROOT} --numof_thread=${NUMOF_THREAD} --thread_num=${i}\
        --numof_classes=${CLASSES} --numof_instances=${INSTANCES}\
        --vertex_num=${VERTEX_NUM} --perlin_min=${PERLIN_MIN} --line_width=${LINE_WIDTH}\
        --radius_min=${RADIUS_MIN} --oval_rate=${OVAL_RATE} --start_pos=${START_POS} &
done
wait
```
