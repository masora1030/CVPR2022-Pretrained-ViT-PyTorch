# ExFractalDB and RCDB

## TOC
- [Summary](#summary)
- [Updates](#updates)
- [Citation](#citation)
- [Requirements](#requirements)
- [Execution Files](#execution-files)
- [ExFractalDB Construction](#exfractaldb-construction-readme)
- [RCDB Construction](#rcdb-construction-readme)
- [Pre-training](#pre-training)
  - [Pre-training with shard dataset](#pre-training-with-shard-dataset)
  - [Pre-trained models](#pre-trained-models)
- [Fine-Tuning](#fine-tuning)
- [Acknowledgements](#acknowledgements)
- [Terms of use](#terms-of-use)


## Summary

The repository contains a Fractal Category Search, ExFractalDB (Extended Fractal DataBase) and RCDB (Radial Contour DataBase) Construction, Pre-training, and Fine-tuning in Python/PyTorch.

<!-- TODO update -->
<!-- The repository is based on the paper:
Hirokatsu Kataoka, Ryo Hayamizu, Ryosuke Yamada, Kodai Nakashima, Sora Takashima, Xinyu Zhang, Edgar Josafat Martinez-Noriega, Nakamasa Inoue and Rio Yokota, "Replacing Labeled Real-Image Datasets With Auto-Generated Contours", IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022 
[[Project](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)] [[PDF (IJCV)](https://link.springer.com/content/pdf/10.1007/s11263-021-01555-8.pdf)] [[PDF (ACCV)](https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf)] [[Dataset](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset)] [[Oral](http://hirokatsukataoka.net/pdf/accv20_kataoka_oral.pdf)] [[Poster](http://hirokatsukataoka.net/pdf/accv20_kataoka_poster.pdf)] -->

## Updates
<!-- TODO update -->
<!-- **Update (Mar 23, 2022)**

* The paper was accepted to International Journal of Computer Vision (IJCV). We updated the scripts and pre-trained models in the extended experiments. [[PDF](https://link.springer.com/content/pdf/10.1007/s11263-021-01555-8.pdf)] [[Pre-trained Models](https://drive.google.com/drive/folders/1tTD-cKKEgBjacCi4ZJ6bRYOv6FsjtGt_?usp=sharing)]


**Update (May 22, 2021)**
* Related project "Can Vision Transformers Learn without Natural Images?" was released. We achieved to train vision transformers (ViT) without natural images. [[Project](https://hirokatsukataoka16.github.io/Vision-Transformers-without-Natural-Images/)] [[PDF](https://arxiv.org/abs/2103.13023)] [[Code](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch)] -->


**Update (May. 21, 2021)**
* Pre-training & Fine-tuning codes
<!-- * Downloadable pre-training models [[Link](https://drive.google.com/drive/folders/1tTD-cKKEgBjacCi4ZJ6bRYOv6FsjtGt_?usp=sharing)] -->
<!-- * Multi-thread preparation with ```param_search/parallel_dir.py``` -->
<!-- * Divide execution files into single-thread processing ```exe.sh``` and multi-thread processing ```exe_parallel.sh``` for FractalDB rendering. -->

## Citation

If you use this code, please cite the following paper:
```bibtex
@InProceedings{Kataoka_2022_CVPR,
    author    = {Kataoka, Hirokatsu and Hayamizu, Ryo and Yamada, Ryosuke and Nakashima, Kodai and Takashima, Sora and Zhang, Xinyu and Martinez-Noriega, Edgar Josafat and Inoue, Nakamasa and Yokota, Rio},
    title     = {Replacing Labeled Real-Image Datasets With Auto-Generated Contours},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21232-21241}
}
``` 

## Requirements

* Python 3.x (worked at 3.7)
* Pytorch 1.x (worked at 1.6.0)
* CUDA (worked at 10.2)
* CuDNN (worked at 8.0)
* OpenMPI (worked at 4.0.5)
* Graphic board (worked at single/four NVIDIA V100)

Please install packages with the following command. (use conda env)

```bash
$ conda env create -f conda_requirements.yaml
$ conda activate cvpr2022_env
```

* Fine-tuning datasets
If you would like to fine-tune on an image dataset, you must prepare conventional or self-defined datasets. To use the following execution file ```scripts/finetune.sh```, you should set the downloaded ImageNet-1k dataset as the following structure.

```misc
/PATH/TO/IMAGENET/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
    ...
  val/
    class1/
      img3.jpeg
      ...
    class2/
      img4.jpeg
      ...
    ...
```

## Execution files

We prepared four execution files in ```exe_scripts``` directory. Please type the following commands on your environment. You can execute ExFractalDB (Extended Fractal DataBase) and RCDB (Radial Contour DataBase) Construction, Pre-training, and Fine-tuning.

- Construct ExFractalDB-1k + Pre-train + Fine-tune

  ```bash
  $ chmod +x exe_scripts/exe_exfractaldb_1k.sh
  $ bash exe_scripts/exe_exfractaldb_1k.sh
  ```

- Construct RCDB-1k + Pre-train + Fine-tune

  ```bash
  $ chmod +x exe_scripts/exe_rcdb_1k.sh
  $ bash exe_scripts/exe_rcdb_1k.sh
  ```

> **Note**
> 
> - ```exe_scripts/exe_exfractaldb_1k.sh``` and ```exe_scripts/exe_rcdb_1k.sh``` are execution scripts under a single node and 4 GPUs environment. For example, if you wish to run the script under a single node and a single GPU, try setting the ```NGPUS``` and ```NPERNODE``` environment variables in the script to 1. ```NGPUS``` means overall the number of GPUs (processes) and ```NPERNODE``` means GPUs (processes) per node.
> 
> - In ```exe_scripts/exe_exfractaldb_21k.sh``` and ```exe_scripts/exe_rcdb_21k.sh```, we use the same configs reported in our paper. Therefore, the setup is multiple nodes and using a large number of GPUs (32 nodes and 128 GPUs for pre-train). If you wish to conduct single-node experiments, please change the config accordingly. Attention, that the number of GPUs used changes the overall batch size proportionally and the optimal learning rate.
> 
> - We use OpenGL to generate ExFractalDB, and because of this, the generating ExFractalDB in remote environment is currently not tested. Please try to generate ExFractalDB in your local environment. Generating RCDB and training models have been tested in remote environment.

<!-- TODO update -->
## ExFractalDB Construction ([README]())
```
$ cd exfractaldb_render
$ bash ExFractalDB_render.sh
```

<!-- TODO update -->
## RCDB Construction ([README](rcdb_render/README.md))
```
$ cd rcdb_render
$ bash RCDB_render.sh
```

## Pre-training

Run the python script ```pretrain.py```, you can pre-train with your dataset.

Basically, you can run the python script ```pretrain.py``` with the following command.

- Example : with deit_base, pre-train ExFractalDB-21k, 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpirun -npernode 4 -np 4 \
      python pretrain.py /PATH/TO/ExFractalDB21000 \
        --model deit_base_patch16_224 --experiment pretrain_deit_base_ExFractalDB21000_1.0e-3 \
        --input-size 3 224 224 \
        --sched cosine_iter --epochs 90 --lr 1.0e-3 --weight-decay 0.05 \
        --batch-size 64 --opt adamw --num-classes 21000 \
        --warmup-epochs 5 --cooldown-epochs 0 \
        --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --remode pixel --interpolation bicubic --hflip 0.0 \
        -j 16 --eval-metric loss \
        --interval_saved_epochs 10 --output ./output/pretrain \
        --log-wandb
    ```

    > **Note**
    > 
    > - ```--batch-size``` means batch size per process. In the above script, for example, you use 4 GPUs (4 process), so overall batch size is 64×4(=256).
    > 
    > - In our paper research, for datasets with more than 10k categories, we basically pre-trained with overall batch size of 8192 (64×128).
    > 
    > - If you wish to distribute pre-train across multiple nodes, the following must be done.
    >   - Set the `MASTER_ADDR` environment variable which is the IP address of the machine in rank 0.
    >   - Set the ```-npernode``` and ```-np``` arguments of ```mpirun``` command.
    >     - ```-npernode``` means GPUs (processes) per node and ```-np``` means overall the number of GPUs (processes).

Or you can run the job script ```scripts/pretrain.sh``` (support multiple nodes training with OpenMPI). Note, the setup is multiple nodes and using a large number of GPUs (32 nodes and 128 GPUs for pre-train).

When running with the script above, please make your dataset structure as following.

```misc
/PATH/TO/ExFractalDB21000/
  cat000000/
    img0_000000_000000_000.png
      ...
  cat000001/
    img0_000001_000000_000.png
      ...
  ...
  cat002099/
    img0_002099_000000_000.png
      ...
```

After above pre-training, trained models are created like ```output/pretrain/pretrain_deit_base_ExFractalDB21000_1.0e-3/model_best.pth.tar``` and ```output/pretrain/pretrain_deit_base_ExFractalDB21000_1.0e-3/last.pth.tar```. 
Moreover, you can resume the training from a checkpoint by setting ```--resume``` parameter.

Please see the script and code files for details on each arguments.

### Pre-training with shard dataset

Shard dataset is also available for accelerating IO processing. 
To make shard dataset, please refer to this repository: https://github.com/webdataset/webdataset. 
Here is an Example of training with shard dataset.

- Example : with deit_base, pre-train ExFractalDB-21k(shard), 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpirun -npernode 4 -np 4 \
      python pretrain.py /NOT/WORKING \
        -w --trainshards /PATH/TO/ExFractalDB21000/SHARDS-{000000..002099}.tar \
        --model deit_base_patch16_224 --experiment pretrain_deit_base_ExFractalDB21000_1.0e-3_shards \
        --input-size 3 224 224 \
        --sched cosine_iter --epochs 90 --lr 1.0e-3 --weight-decay 0.05 \
        --batch-size 64 --opt adamw --num-classes 21000 \
        --warmup-epochs 5 --cooldown-epochs 0 \
        --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --remode pixel --interpolation bicubic --hflip 0.0 \
        -j 1 --eval-metric loss --no-prefetcher \
        --interval_saved_epochs 10 --output ./output/pretrain \
        --log-wandb
    ```
​
When running with the script above with shard dataset, please make your shard dataset structure as following.

```misc
/PATH/TO/ExFractalDB21000/
    SHARDS-000000.tar
    SHARDS-000001.tar
    ...
    SHARDS-002099.tar
```

### Pre-trained models

Our pre-trained models are available in this [[Link](https://drive.google.com/drive/folders/1ikNUxJoMCx3Lx2TMrXfLdIwI6wwK5w_W?usp=sharing)].

We have mainly prepared two different pre-trained models. These pre-trained models are trained on {ExFractalDB, RCDB}-21k.

```misc
exfractal_21k_base.pth.tar: --model deit_base_patch16_224 --experiment pretrain_deit_base_ExFractalDB21000_1.0e-3_shards
rcdb_21k_base.pth.tar: --model deit_base_patch16_224 --experiment pretrain_deit_base_RCDB21000_1.0e-3_shards
```

If you would like to additionally train from the pre-trained model, please command with the next fine-tuning code as follows.

```bash
# exfractal_21k_base.pth.tar
$ mpirun -npernode 4 -np 4 \
  python finetune.py /PATH/TO/YOUR_FT_DATASET \
    --model deit_base_patch16_224 --experiment finetune_deit_base_YOUR_FT_DATASET_from_ExFractalDB21000_1.0e-3_shards \
    --input-size 3 224 224 --num-classes YOUR_FT_DATASET_CATEGORY_SIZE \
    --output ./output/finetune \
    --log-wandb \
    --pretrained-path /PATH/TO/exfractal_21k_base.pth.tar

# rcdb_21k_base.pth.tar
$ mpirun -npernode 4 -np 4 \
  python finetune.py /PATH/TO/YOUR_FT_DATASET \
    --model deit_base_patch16_224 --experiment finetune_deit_base_YOUR_FT_DATASET_from_RCDB21000_1.0e-3_shards \
    --input-size 3 224 224 --num-classes YOUR_FT_DATASET_CATEGORY_SIZE \
    --output ./output/finetune \
    --log-wandb \
    --pretrained-path /PATH/TO/rcdb_21k_base.pth.tar
```

## Fine-tuning

Run the python script ```finetune.py```, you additionally train other datasets from your pre-trained model.

In order to use the fine-tuning code, you must prepare a fine-tuning dataset (e.g., ImageNet-1k, CIFAR-10/100, Pascal VOC 2012). Please look at [Requirements](#Requirements) for a dataset preparation.

Basically, you can run the python script ```finetune.py``` with the following command.

- Example : with deit_base, fine-tune ImageNet-1k from pre-trained model (with ExFractalDB-21k), 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpirun -npernode 4 -np 4 \
      python finetune.py /PATH/TO/IMAGENET \
        --model deit_base_patch16_224 --experiment finetune_deit_base_ImageNet1k_from_ExFractalDB21000_1.0e-3 \
        --input-size 3 224 224 --num-classes 1000 \
        --sched cosine_iter --epochs 300 --lr 1.0e-3 --weight-decay 0.05 \
        --batch-size 64 --opt adamw \
        --warmup-epochs 5 --cooldown-epochs 0 \
        --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 \
        --drop-path 0.1 --reprob 0.25 -j 16 \
        --output ./output/finetune \
        --log-wandb \
        --pretrained-path ./output/pretrain/pretrain_deit_base_ExFractalDB21000_1.0e-3/model_best.pth.tar
    ```

Or you can run the job script ```scripts/finetune.sh``` (support multiple nodes training with OpenMPI).

Please see the script and code files for details on each arguments.

## Acknowledgements

Training codes are inspired by [timm](https://github.com/rwightman/pytorch-image-models) and [DeiT](https://github.com/facebookresearch/deit).

## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST) and Tokyo Institute of Technology (TITech) are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the images and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.