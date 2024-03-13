# PiRO 

## Requirements
Please clone this repo and install the dependencies using:
```bash
conda env create -f environment_piro.yml
```

## Datasets
The datasets are organized such that the multi-view images of each object identity are stored in a separate subfolder. 
The train and splits for the ObjectPI, ModelNet-40, and FG3D datasets can be downloaded from [Google Drive]() .
The mapping of object-identity to categories is also provided as an `obj2cat.npy` file.

## Training

## Testing and Evaluation
Download the model weights from [Google Drive]()

For ObjectPI dataset: 
```bash
python testPiRO.py OOWL dual model_weights/OOWL/Dual_CAT_PiOBJ_PiCAT.pth
```
For ModelNet-40 dataset:
```bash
python testPiRO.py MNet40 dual model_weights/MNet40/Dual_CAT_PiOBJ_PiCAT.pth
```
For FG3D dataset:
```bash
python testPiRO.py FG3D dual model_weights/FG3D/Dual_CAT_PiOBJ_PiCAT.pth
```
## Acknowledgement
