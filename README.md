# Hangul-SSC
Single Shot Classifier for Hangul Classfication  
Forked from [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) by amdegroot

## Environment
- CUDA toolkit 11.0
- Pytorch 1.7.0
- SciPy 1.5.3
- Numpy 1.19.2
- imgaug 0.4.0

## Dataset
We used HIL-SERI, which is intersection of HIL and SERI95 dataset. HIL dataset is only accessible via [EIRIC](https://www.eiric.or.kr/special/special.php) website, and you can get SERI95 dataset from [HangulDB](https://github.com/callee2006/HangulDB) repository. We took only 128 classes on them, and took some preprocessing.
* Note: To get HIL dataset, you must wrote memorandum to only use dataset on reseaching purpose! By this reason, we **do not** provide dataset.

## Training
- With plenty data, you can start your own training. 
- First, you have to download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- After download, place weight file on root directory of this project.
- You now can train SSC model with `train.py` file. Check argument for detail.

## Evaluation
- To evaluate trained model, you can use `eval.py` or `eval_list.py`.
- `eval.py` is for *single model file*. you have to put model file's directory.
- `eval_list.py` is for *multiple model files*. you have to put directory which contains model files.

## Accuracy
**On preprocessed SERI-95 dataset, SSC scored accuracy of 98.56%.**
