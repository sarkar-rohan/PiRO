# Dual Pose-Invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval

## Requirements
Please clone this repo and install the dependencies using:
```bash
conda env create -f environment_piro.yml
```

## Datasets
We use the following multi-view datasets in our work:
- ObjectPI (also known as Objects in the Wild or OOWL) [1]
- ModelNet-40 [2]
- FG3D [3] 

For learning, we have organized these datasets such that the multi-view images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The train and test splits for the above-mentioned datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1BEl7XAqYK13NGOMuahMy-hxK4oSLRc8J/view?usp=sharing) .
The mapping of object-identities to categories is also provided as `train_o2c.npy` and `test_o2c.npy` files.

For using these datasets please unzip the data.zip file using
```bash
unzip data.zip
```

## Training
If you wish to train a model from scratch, please follow the instructions below: 

To train the dual-encoder model, please use the following command: 
```bash
python trainPiRO_Dual.py dataset_name experiment_name run_number
```
To train the single-encoder model, please use the following command: 
```bash
python trainPiRO_Single.py dataset_name experiment_name run_number
```
where, 
- `dataset_name` should be passed as `OOWL` for ObjectPI, `MNet40` for ModelNet-40, and `FG3D` for the FG3D datasets.
- `experiment_name` is a user-specified string for saving the trained model weights in the result directory.
- `run_number` is an integer for different runs

For example, to train on the ObjectPI (OOWL) dataset, please run the following:
```bash
python trainPiRO_Dual.py OOWL R1 1
```
## Testing and Evaluation

### Evaluation of our trained models
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1jsJaTmgF7M8Gvh0yIDGr9tWch339qMKE?usp=sharing)

For ObjectPI dataset: 
```bash
python testPiRO.py OOWL dual model_weights/OOWL/Dual_CAT_PiOBJ_PiCAT.pth
```
![ObjectPI](https://github.com/sarkar-rohan/PiRO/assets/17092235/ac40bbc0-9504-46f2-967f-348aca0632d2)

For ModelNet-40 dataset:
```bash
python testPiRO.py MNet40 dual model_weights/MNet40/Dual_CAT_PiOBJ_PiCAT.pth
```
![ModelNet-40](https://github.com/sarkar-rohan/PiRO/assets/17092235/f5259666-cade-4fff-bae6-8691585f090b)

For FG3D dataset:
```bash
python testPiRO.py FG3D dual model_weights/FG3D/Dual_CAT_PiOBJ_PiCAT.pth
```
![FG3D](https://github.com/sarkar-rohan/PiRO/assets/17092235/20038525-928a-4884-9d08-19addc84aa3f)


### Evaluation of models trained from scratch
If you trained a model from scratch it will be stored in the results directory
To evaluate the dual-encoder model, please use the following command: 
```bash
python testPiRO.py dataset_name dual model_path
```
To evaluate the single-encoder model, please use the following command: 
```bash
python testPiRO.py dataset_name single model_path
```

## References
[1] Chih-Hui Ho, Pedro Morgado, Amir Persekian, and Nuno Vasconcelos. PIEs: Pose invariant embeddings. In Computer Vision and Pattern Recognition (CVPR), 2019.  
[2] Zhirong Wu, S. Song, A. Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and J. Xiao. 3D shapenets: A deep representation for volumetric shapes. In Computer Vision and Pattern Recognition (CVPR), pages 1912–1920, Los Alamitos, CA, USA, 2015.  
[3] Xinhai Liu, Zhizhong Han, Yu-Shen Liu, and Matthias Zwicker. Fine-grained 3D shape classification with hierarchical part-view attentions. IEEE Transactions on Image Processing, 2021.  

## Citation
If you use this method in your research, please cite :  
```
@InProceedings{Sarkar_2024_CVPR,  
    author    = {Sarkar, Rohan and Kak, Avinash},  
    title     = {Dual Pose-invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval},  
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
    month     = {June},  
    year      = {2024},  
    pages     = {17077-17085}  
}
```
