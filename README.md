A simple 1D Convolutional Neural Network (1D-CNN) implemented in PyTorch for ECG signal classification using the PTB-XL dataset.
The model includes three convolutional layers and is trained for 10 epochs by default to run efficiently on local machines but can easily be extended with more layers and higher epochs.
The project requires the correct environment setup, which can be done using either the pyproject.toml file with uv sync or the requirements.txt file with pip install -r requirements.txt.
The main notebook, simple_1dcnn.ipynb, contains the PyTorch model and training pipeline, while ptbxl_utils.py handles dataset configuration, fold selection (folds 1–10), and helper functions.
Two dataset modes are supported: filtered mode (AFIB vs NORM) and unfiltered mode (AFIB vs ALL other classes), along with a choice of ECG signal frequency (100Hz or 500Hz).
To run the project, download the PTB-XL dataset (v1.0.3) from PhysioNet
, and ensure the folders records100 and records500, along with the files ptbxl_database.csv and scp_statements.csv, are placed correctly.
Then, configure data paths, folds, and training settings in the notebook—the code will automatically detect whether to use CPU or GPU.
An additional notebook, exploreDataset.ipynb, is included for dataset exploration.