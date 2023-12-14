# Document Explanation

## Input data
`\images\df.csv`: detailed information for each timepoint of each subject
`\images\SimulatedData__Reconstruction__starman__subject_s0__tp_0.npy`: numpy array data for subject 0 timepoint 0

## Model architecture and data preprocessing
`\lib\UOMETM_model.py`: an autoencoder framework consists of convolutional networks and mixed-effects modeling in the latent space
`\lib\dataset.py`: processing Starmen dataset into training and test datasets in a 5-fold manner

## Implementation code
`\training\main.py`

## Trained model storation
`\trained_model\0_fold_UOMETM`: trained model for the first fold

## Training process visualization
`\training\visualization`

`\training_process\loss.png`: convergence of all losses during training
`\training_process\reconstruction.png`: visualization of reconstruction quality
`\distribution_latent_space\Z_distribution.png`: distribution of all dimensions of latent space $\mathcal{Z}$
`\visualize_latent_space\simulation_Z.png`: visualization of all dimensions of latent space $\mathcal{Z}$
`\visualize_latent_space\subtraction_simulation_Z.png`: subtraction between adjacent images to show limb progression

## Analysis after training
`\post_analysis\visualization`

`\Z.png`, `\ZU.png`, `\ZV$.png`: visualize representation spaces $\mathcal{Z}$, $\mathcal{Z}^\mathbf{U}$ and $\mathcal{Z}^\mathbf{V}$
`\extrapolation.png`: extrapolation of individual trajectory
