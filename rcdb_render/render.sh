save_root="./rcdb"
numof_category=1000
numof_instances=1000
vertex_num=200
perlin_min=0
line_width=0.1
radius_min=0
oval_rate=2
start_pos=400
numof_thread=40

# Multi-thread processing
for ((i=0 ; i<${numof_thread} ; i++))
do
    python rcdb_render/make_rcdb.py --save_root=${save_root} --numof_thread=${numof_thread} --thread_num=${i}\
        --numof_classes=${numof_category} --numof_instances=${numof_instances}\
        --vertex_num=${vertex_num} --perlin_min=${perlin_min} --line_width=${line_width}\
        --radius_min=${radius_min} --oval_rate=${oval_rate} --start_pos=${start_pos} &
done
wait