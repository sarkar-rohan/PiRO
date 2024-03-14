# Dual Pose-Invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval

## Requirements
Please clone this repo and install the dependencies using:
```bash
conda env create -f environment_piro.yml
```

## Datasets
We use the following multi-view datasets in our work:
- ObjectPI (also known as Objects in the Wild or OOWL)
- ModelNet-40
- FG3D

For learning, we have organized these datasets such that the multi-view images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The train and test splits for the above-mentioned datasets can be downloaded from [Google Drive]() .
The mapping of object-identities to categories is also provided as `train_o2c.npy` and `test_o2c.npy` files.

For using these datasets please unzip the data.zip file using
```bash
unzip data.zip
```

## Training
If you wish to train a model from scratch
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

## Testing and Evaluation

### Evaluation of our trained models
Download the model weights from [Google Drive]()

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
## Acknowledgement
