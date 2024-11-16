
# ML models for compound flood simulation

## Description
Comparison of PINN and data-driven models for simulating compound river and coastal flooding

## Installation and Requirements

### PINN (Physics-Informed Neural Networks)
- **TensorFlow Version**: Requires TensorFlow 1.14.0
- **Environment Setup**: Use the environment details provided in `requirement_tf1.txt` to set up your Conda environment.
```bash
conda create --name tf1 --file requirement_tf1.txt
conda activate tf1
```

### Data-driven Model
- **TensorFlow Version**: Requires TensorFlow 2.17.0
- **Environment Setup**: Use the environment details provided in `requirement_tf2.txt` to set up your Conda environment.
```bash
conda create --name tf2 --file requirement_tf2.txt
conda activate tf2
```

## Usage

### Before training
Before running the code, need to create folders to save the model output
For CNN, create /files/CNN
for PINN, create /saved_model

### Training and Results

#### PINN
- **Training**: To train the model, run:
```bash
python PINN_test_bnd_uh_Telemac.py
```
- **Result Plotting and Comparison**: For plotting and comparing results, use:
```bash
python PINN_plot_comparison.py
```

#### Data-driven Model
- **CNN Training**: To train the CNN model, execute:
```bash
python train_CNN.py
```
- **Result Visualization**: To visualize the results of the CNN model, run:
```bash
python predict_CNN.py
```
