# ======== parameter for pre-trained model ========
# model size
MODEL=base
# initial learning rate for pre-train
PRE_LR=1.0e-3
# name of dataset for pre-train
PRE_DATA_NAME=ExFractalDB
# num of classes for pre-train
PRE_CLASSES=21000
# path to checkpoint of pre-trained model
CP_PATH=./output/pretrain/pretrain_deit_${MODEL}_${PRE_DATA_NAME}${PRE_CLASSES}_${PRE_LR}_shards/model_best.pth.tar

# ======== parameter for fine-tuning ========
# output dir path
OUT_DIR=./output/finetune
# path to fine-tune dataset
SOURCE_DATASET_DIR=/PATH/TO/IMAGENET/
# name of dataset
DATA_NAME=ImageNet1k
# initial learning rate
LR=1.0e-3
# num of classes
CLASSES=1000
# num of epochs
EPOCHS=300
# num of GPUs
NGPUS=16
# num of processes per node
NPERNODE=4
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

mpirun -npernode $NPERNODE -np $NGPUS \
python finetune.py ${SOURCE_DATASET_DIR} \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_${DATA_NAME}_from_${PRE_DATA_NAME}${PRE_CLASSES}_${PRE_LR} \
    --input-size 3 224 224 --num-classes ${CLASSES} \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BS} --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ${CP_PATH}
