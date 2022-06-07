SAVE_ROOT="./dataset/rcdb"
CLASSES=1000
INSTANCES=1000
VERTEX_NUM=200
PERLIN_MIN=0
LINE_WIDTH=0.1
RADIUS_MIN=0
OVAL_RATE=2
START_POS=400
NUMOF_THREAD=40

# Multi-thread processing
for ((i=0 ; i<${NUMOF_THREAD} ; i++))
do
    python rcdb_render/make_rcdb.py --save_root=${SAVE_ROOT} --numof_thread=${NUMOF_THREAD} --thread_num=${i}\
        --numof_classes=${CLASSES} --numof_instances=${INSTANCES}\
        --vertex_num=${VERTEX_NUM} --perlin_min=${PERLIN_MIN} --line_width=${LINE_WIDTH}\
        --radius_min=${RADIUS_MIN} --oval_rate=${OVAL_RATE} --start_pos=${START_POS} &
done
wait

# pretrain

# model size
MODEL=base
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=RCDB
# num of epochs
EPOCHS=300
# path to train dataset
SOURCE_DATASET=${SAVE_ROOT}
# output dir path
OUT_DIR=./cheak_points/${MODEL}/${CLASSES}/pretrain
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python pretrain.py ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR} \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 16 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb

# finetune

# ======== parameter for pre-trained model ========
# initial learning rate for pre-train
PRE_LR=1.0e-3
# name of dataset for pre-train
PRE_DATA_NAME=RCDB
# num of classes for pre-train
PRE_CLASSES=1000
# path to checkpoint of pre-trained model
CP_PATH=${OUT_DIR}/pretrain_deit_${MODEL}_${PRE_DATA_NAME}${PRE_CLASSES}_${PRE_LR}/model_best.pth.tar

# ======== parameter for fine-tuning ========
# output dir path
OUT_DIR=./cheak_points/${MODEL}/${CLASSES}/finetune
# path to fine-tune dataset
SOURCE_DATASET_DIR=/PATH/TO/IMAGENET
# name of dataset
DATA_NAME=ImageNet
# initial learning rate
LR=1.0e-3
# num of classes
CLASSES=1000
# num of epochs
EPOCHS=300
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python finetune.py ${SOURCE_DATASET_DIR} \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_${DATA_NAME}_from_${PRE_DATA_NAME}${PRE_CLASSES}_${PRE_LR} \
    --input-size 3 224 224 --num-classes ${CLASSES} \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    -j 16 \
    --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ${CP_PATH}
