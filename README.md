### Example results

<img width="3021" height="1530" alt="Fig  1  Traditional fixed-length vs  our parser-guided attribute learning (PAL) architectures  Zoom in for better clarity" src="https://github.com/user-attachments/assets/2e1eb0f1-3778-4caa-a4ac-9d72c55cac16" />

<img width="3305" height="1905" alt="Fig  2  Overall architecture of GSF-GAN  Zoom in for better clarity" src="https://github.com/user-attachments/assets/ba82a52b-0b72-49ff-ade6-fbda0020e75b" />

<img width="2341" height="1590" alt="Fig  7   Examples of images synthesized by Patri  11 , LMD  13 , GALIP  15 , CLIP-GAN  16 , and our proposed GSF-GAN, which is c" src="https://github.com/user-attachments/assets/6265a478-6128-4b1f-93de-c623f76593f8" />


## Refinement-Guided Text-to-Image Synthesis via Global Semantic Fusion GAN. 

we propose a refinement-based framework designed as the "Global Semantic Fusion Generative Adversarial Network (GSF-GAN)". The fidelity and consistency of generated images are better matched through three key components.  

### Implementation 

### How to use them

0. **Requirements** 

```
Python >= 3.9
PyTorch >= 1.9
NVIDIA GPU + CUDA cuDNN
```

1. **Data** 
   1. Download metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to your path.
   2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data.
   3. Download [coco](http://cocodataset.org/#download) dataset.
   4. extract the source data to your path.

2. **Pretrained Models**
   1. the [models file](https://pan.baidu.com/s/1-V2Mp0wmX_tQxl6mOtnKpw) of our GSF-GAN which obtain the best performance. CODE:zrE2.
   2. later experiments found that good results often occur between 550 and 650 epoches, we suggest you choose the model in this scope.
3. Training 
   * Modify the parameters in parse to your local path
   * You can modify some parameters in the cfg file

```python
python /GSF_GAN/code/main.py --cfg cfg/bird_DMGAN.yml --gpu 0
python /GSF_GAN/code/main.py --cfg cfg/coco_DMGAN.yml --gpu 0
```


4. Validation

```python
# image genaration:
python /GSF_GAN/code/main.py --cfg cfg/eval_bird.yml --gpu 0
python /GSF_GAN/code/main.py --cfg cfg/eval_coco.yml --gpu 0

# FID
python fid_score.py --gpu 0 --batch-size 50 --path1 bird_val.npz --path2 your generated picture path
# R-precision
# You will get the result of R-precision when 30k pictures are generated
python /GSF_GAN/code/main.py --cfg cfg/eval_bird.yml --gpu 0
python /GSF_GAN/code/main.py --cfg cfg/eval_coco.yml --gpu 0

```



### Acknowlegement
The pre-process data and code borrows heavily from
- [Recurrent Affine Transformation for Text-to-image Synthesis](https://arxiv.org/abs/2204.10482) [[code]](https://github.com/senmaoy/RAT-GAN) 
- (https://github.com/MinfengZhu/DM-GAN),
-(https://github.com/xueqinxiang/DMF-GAN),
we apprecite the authors for sharing their codes and data.
