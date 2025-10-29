A simple 1D Convolutional Neural Network (1D-CNN) implemented in PyTorch for ECG signal classification using the PTB-XL dataset. 
The model includes three convolutional layers and is trained for 10 epochs by default to run efficiently on local machines,
but it can easily be extended with more layers and higher epochs. The project requires the correct environment setup, which can be
done using either the pyproject.toml file with uv sync or the requirements.txt file with pip install -r requirements.txt. The main notebook,
simple_1dcnn.ipynb, contains the PyTorch model and training pipeline, while ptbxl_utils.py handles dataset configuration and helper functions. 
To run the project, ensure the PTB-XL dataset folders records100 and records500 are placed correctly, configure data paths and training settings
in the notebook, and the code will automatically detect whether to use CPU or GPU. An additional notebook, exploreDataset.ipynb, is included for dataset exploration.
