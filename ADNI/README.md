# Document Explanation

## Model architecture and data preprocessing
`/lib/UOMETM_model.py`: an autoencoder framework consists of FullyConnected layers and mixed-effects modeling in the latent space
`/lib/dataset.py`: processing ADNI dataset into training and test datasets

## Implementation code
`/training/main.py`

## Analysis after training
`/visualize_ZU`: visualization of cortical thickness captured in the space $\mathcal{Z}^\mathbf{U}$ on a cortical surface

`drawsurface_avg.m`: cortical surface drawing code

`visualize_ZU.mat`: cortical thickness data including

year_$age$_$left\&right$ (3x163842): left or right hemisphere cortical thickness data for three diagnostic groups (each row represents CN, MCI, AD) at the $age=60, 68, 77, 85$.
