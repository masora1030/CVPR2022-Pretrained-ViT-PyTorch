# model size
MODEL=base
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=ExFractalDB
# num of classes
CLASSES=21000
# num of epochs
EPOCHS=90
# path to shard train dataset
SOURCE_TAR_DATASET="/PATH/TO/ExFractalDB21000/SHARDS-{000000..002099}.tar"
# output dir path
OUT_DIR=./output/pretrain
# num of GPUs
NGPUS=128
# num of processes per node
NPERNODE=4
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

mpirun -npernode $NPERNODE -np $NGPUS \
python pretrain.py /NOT/WORKING \
    -w --trainshards ${SOURCE_TAR_DATASET} \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_shards \
    --input-size 3 224 224 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw --num-classes ${CLASSES} \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb
