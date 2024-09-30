# DTSS（Domain Transformation and Semantic Segmentation）

This is the official repository for "A Novel Workflow of Segmentation for Finer Mineral Distingished ：AttentionGAN-Swin-Transformer Fused Network". Please cite this work if you find this repository useful for your project.


We proposed a workflow - DTSS (Domain Transformation and Semantic Segmentation): first use AttentionGAN to convert the CT image domain to the (SEM) scanning electron microscope image domain, and then use Swin Transformer to perform image segmentation. By introducing attention masks and content masks, AttentionGAN can more effectively learn the mapping relationship between the two domains, thereby generating images in the corresponding target domain. On the basis of domain transformation, we further use Swin-Transformer for image segmentation. Swin-Transformer is a Transformer-based model that efficiently processes image data through a self-attention mechanism. Compared with traditional convolutional neural networks (CNN), Swin-Transformer's global receptive field and stronger modeling capabilities give it significant advantages when processing complex, multi-mineral rock images. Swin-Transformer is able to capture long-range dependencies in images, which is particularly important for identifying and segmenting mineral dependencies in rocks.


## Usage

### AttentionGAN

#### dataset

```bash
sh ./datasets/download_cyclegan_dataset.sh dataset_name
```

#### Training

```bash
python train.py --dataroot ./datasets/yourdatasets/ --name yourdatasets_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 100 --niter_decay 100 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100
```

#### Inference

```bash
python test.py --dataroot ./datasets/yourdatasets/ --name yourdatasets_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest
```


### SwinTransformer

#### Training

To train with pre-trained models, run:

```bash
#single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

For example, to train an UPerNet model with a Swin-T backbone and 8 gpus, run:

```bash
python tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL> 
```