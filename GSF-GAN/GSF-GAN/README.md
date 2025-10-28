### Example results

![GSF-GAN](https://github.com/user-attachments/assets/1a3a62a8-8f04-4e3f-b7c2-641026ed33df)

![Feature Matching Loss](https://github.com/user-attachments/assets/1a709da4-ea71-49a1-b5f1-48b52a41bb90)

![IMAGE-TEXT  MATCHING LOSS](https://github.com/user-attachments/assets/ce8aec5f-90f6-4bcc-a1c5-55abfcf57d4e)

![ChatGPT Image May 30, 2025, 10_57_37 PM](https://github.com/user-attachments/assets/25179902-7b6d-4a1d-a0ba-bce4d2789e00)


##  Global Semantic Fusion Generative Adversarial Network (GSF-GAN). 

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
