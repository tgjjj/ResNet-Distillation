# ResNet with No Res
##### Accelerating ResNet by removing its residual branch

--------------------------------------------------------

### Introduction
ResNet is a commonly used backbone network for many vision tasks.  Its superior performance come from its residual structure, which solved the vanishing gradient problem. Although residual branches are benificial during training, **they result in additional time cost during forward inferencing**. Our Experiments showed that these residual branches **can be removed using knowledge distillation**, resulting in a rather **small accuracy reduction while accelerating the network** during inference. Similar methods can be applied to any network with branches like ResNet (*Inception series, for example*) and might be useful (*need further experiments to prove this*).

--------------------------------------------------------

### About This Project
#### Libraries Required
>CUDA = 10.1 (optional but recommended)  
>Python = 3.8.3  
>Numpy = 1.18.1  
>Pytorch = 1.5.0  
>TensorboardX = 2.2.1 & Tensorflow = 2.2.0 (optional)  

All libraries were installed using *Anaconda2* except *Tensorflow*, which was installed using *Pip*.   
*Numpy* and *CUDA* will be installed automatically while installing *Pytorch*. Try `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`  
*TensorboardX* is used for visualizing training losses and accuracies, and *Tensorflow* is installed to support *TensorboardX*.  
Other libraries were installed by *Anaconda2* automatically.  
Other versions of these libraries might also work for you. But I can't guarantee that.  
#### Run a simple test demo
Use the following command to run a simple test demo.   
The two models in *{$project_root_dir}/models/ResNet-20/* will be tested on *CIFAR100* dataset.  
`cd {$project_root_dir}`  
`python ResNet_Distillation.py`  
#### Project Files
This project consists of three python scripts.  
>ResNet_Distillation.py  
>ResNet.py  
>ResNet_plain.py  

*ResNet_Distillation.py* is the main script. It contains dataset loading, train settings, test settings, etc.  
*ResNet.py* defines a set of **standard ResNets** modified according to the ResNet paper to suit the CIFAR100 dataset.  
*ResNet_plain.py* defines a corresponding set of **ResNets without residual branches**, "plain ResNet" for short.  
The latter two scripts are modified based on the [official ResNet implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)provided by *Pytorch*.

### Training method
We used a 2-stage training method to obtain a plain ResNet.  
In the first stage, we train the stardard ResNet on CIFAR100.  
In the second stage, we use the model from the first stage to **supervise** the training of the plain ResNet. To be more specific, we divide the plain ResNet natually into **3 stages** (by the downsampling layer), and we train them **one by one**. For each stage, **we train to make its output feacher map as similar to the standard ResNet's output as possible**. To measure the similarity between two output feature maps, we use a per pixel loss, which is the **manhattan distance** between the two tensors. Each loss is responsible for training **the one stage just before it**. Finally, after the three stages are well trained, we fine-tune the whole plain ResNet for a few epoches. Refer to the image below for a better understanding of this procedure.

### Experiment Results
Results|ResNet-18|ResNet-18-Plain|ResNet-20|ResNet-20-Plain|
:-----:|:-------:|:-------------:|:-------:|:-------------:|
Accuracy|48.07%|48.49%|59.98%|57.67%|
Inference Time(s)|1.119|0.818|0.218|0.198|
ResNet-18 is the original implementation for ImageNet, which has 4 stages and 64-256 channels.
ResNet-20 is specially designed for CIFAR dataset, which has only 3 stages and 16-64 channels. That's why it's faster than ResNet-18.
The inference time is the total inference time on the CIFAR100 test set, with batch size = 1024 for ResNet-20.
I also tried ResNet-101 with 3 stages, but it seems that 3 stages is not enough for such deep network. The result is either not converging or getting a rather low accuracy, which is probably the results of the final fine-tune procedure in my point of view.

### Special Thanks
Special thanks to my senior [Xin Ye](https://github.com/NixeyJay) for the basic idea of this project and his pungent advices.  
Special thanks to [my college](http://www.cbeis.zju.edu.cn/cbeiscn/main.htm) for providing all the resources and experimental environment in this project.  
