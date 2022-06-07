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