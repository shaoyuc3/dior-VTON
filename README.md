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
Download the __VITON Dataset__ from [here](https://drive.google.com/file/d/11kKsTXoRwfMzx32I6OADJYPmlRLxpRv8/view?usp=sharing).

### Custom dataset/from scratch
For custom dataset or train from scratch, please generate the data folder with the same structure as below.

1. human parse: [Human Parser](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) (with LIP labels)
2. pose: [Openpose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
3. dense pose: [DensePose](https://github.com/facebookresearch/DensePose)


## Run Demo
You can download the pretrained weights from [here](https://drive.google.com/drive/folders/1Qgaj4n9e412CDdNQOWiuIuEy8Ysx2HrY) and unzip it at ```checkpoints/```.

After downloading the pretrained model and setting the data, you can try out the applications in notebook [demo.ipynb](demo.ipynb).


## Training
__Parser-Based Appearance Flow Style__

To warm up the Parser-Based Appearance Flow Field Estimator, first download the vgg checkpoint from [here](https://github.com/senhe/flow-style-vton) and put the checkpoint under the folder ```models/networks/flowStyle```.

Then run
```
sh scripts/run_PBAFN.sh
```

__Note__: if you don't want to warmup the Estimator, you can extract the weights from [here](https://drive.google.com/drive/folders/1upRRswJf_hXldl48w5QCX7LOJp71otJB).

__Training__
After warming up the flownet, train the pipeline by running
```
sh scripts/run_train.sh
```

## Evaluations
To evaluate (SSIM, FID and LPIPS) and generate images, run
```
sh scripts/run_eval.sh
```

Our generated images [\[256x192\]](https://drive.google.com/drive/folders/16RHLVxGx7kYD_YZ7MoSOlWO2TnHHXYVU).

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
