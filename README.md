# Dressing in Order (DiOr) on VITON Dataset

This is an implementation of __Dressing in Order__ on VITON dataset using __Parser-Based Appearance Flow Style__.


![](Images/short_try_on_editing.png)

__More results__

Play with [demo.ipynb](demo.ipynb)!

## Requirements

- python 3.6.13
- torch 1.10.1
- torchvision 0.11.2
- opencv
- visdom

## Get Started
Run
```
pip install -r requirements.txt
```

__If one wants to run inference only:__
Please specify ```--frozen_flownet```.

## Dataset
Download the __VITON Dataset__ from [here](https://drive.google.com/file/d/1TuRdpNj-SeUBwLg_mm6RbR7SCioscK2s/view?usp=sharing).


After the processing, you should have the dataset folder formatted like:
```
+ $DATA_ROOT
|   + train (all training images)
|   |   - xxx.jpg
|   |     ...
|   + trainM_lip (human parse of all training images)
|   |   - xxx.png
|   |     ...
|   + test (all test images)
|   |   - xxx.jpg
|   |     ...
|   + testM_lip (human parse of all test images)
|   |   - xxx.png
|   |     ...
|   - fashion-pairs-train.csv (paired poses for training)
|   - fashion-pairs-test.csv (paired poses for test)
|   - fashion-annotation-train.csv (keypoints for training images)
|   - fashion-annotation-test.csv  (keypoints for test images)
|   - train.lst
|   - test.lst
|   - standard_test_anns.txt
```


---

## Run Demo
Please download the pretrained weights from [here](https://drive.google.com/drive/folders/1-7DxUvcrC3cvQV67Z2QhRdi-9PMDC8w9?usp=sharing) and unzip at ```checkpoints/```. 

After downloading the pretrained model and setting the data, you can try out our applications in notebook [demo.ipynb](demo.ipynb).

*(The checkpoints above are reproduced, so there could be slightly difference in quantitative evaluation from the reported results. To get the original results, please check our released generated images [here](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing).)*

*(```DIORv1_64``` was trained with a minor difference in code, but it may give better visual results in some applications. If one wants to try it, specify ```--netG diorv1```.)*

---
## Training

__Warmup the Global Flow Field Estimator__

Note, if you don't want to warmup the Global Flow Field Estimator, you can extract its weights from GFLA by downloading the pretrained weights GFLA from [here](https://github.com/RenYurui/Global-Flow-Local-Attention). (Check Issue [#23](https://github.com/cuiaiyu/dressing-in-order/issues/23) for how to extract weights from GFLA.)

Otherwise, run

```
sh scripts/run_pose.sh
```

__Training__

After warming up the flownet, train the pipeline by
```
sh scripts/run_train.sh
```
Run ```tensorboard --logdir checkpoints/$EXP_NAME/train``` to check tensorboard.

*Note: Resetting discriminators may help training when it stucks at local minimals.*


## Evaluations

__Download Generated Images__ 

Here are our generated images which are used for the evaluation reported in the paper. (Deepfashion Dataset) 
- [\[256x176\]](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing)
- [\[256x256\]](https://drive.google.com/drive/folders/1GOQVMhBKvANKutLDbzPbE-Zrb6ai9Eo8?usp=sharing)

__SSIM, FID and LPIPS__

To run evaluation (SSIM, FID and LPIPS) on pose transfer task: 
```
sh scripts/run_eval.sh
```

---
## Cite us!
If you find this work is helpful, please consider starring :star2: this repo and cite us as
```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Aiyu and McKee, Daniel and Lazebnik, Svetlana},
    title     = {Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-On and Outfit Editing},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14638-14647}
}
```
## Acknowledgements
This repository is built up on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention),
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), 
[PATN](https://github.com/tengteng95/Pose-Transfer) and 
[MUNIT](https://github.com/NVlabs/MUNIT). Please be aware of their licenses when using the code. 

Thanks a lot for the great work to the pioneer researchers!
