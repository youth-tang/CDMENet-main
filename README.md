# CDMENet-main
This is the code for the paper "Semi-supervised Counting of Grape Berries in the Field Based on Density Mutual Exclusion" 
## Prepare 
  1.1 Datasets can Found in: 
  
  github：Embrapa Wine Grape Instance Segmentation Dataset – Embrapa WGISD
  
  arxiv：Grape detection, segmentation and tracking using deep neural networks and three-dimensional association
  
  youtube：Grape detection, segmentation and tracking

  1.2 Setting Runing Environment：
  
  Ubuntu 20.04
  
  Intel Core i9-10900X CPU@3.70GHz
  
  python 3.8
  
  Pytorch  1.7.1

  GeForce RTX 3090

## Data Processing:
  follow the file "make_dataset.py" to produce the ground-truth density map, file.mat to file.h5
  
## Training the model:
  python train.py train.json val.json 0 0 to train your model
  
## Testing the model:
  python val.py 
  
  Notice the path of all files in these codes, you should modify them to suit your condition.
  
