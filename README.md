A simple 1D-CNN implemented in pyTorch. I created only 3 layers ( not much deep) and set epoch on only 10 so it can work on my machine.
With a strong PC with a strong GPU, can have epoch opp til 100 and more layers(10 for example). More deeper, sure better.
to run the code on local machine in vs code: 
1. copy the 2 folder records100 and records500 to the folder ptbxl_dataset-1.o.3
2. important to have the correct invironment, i have 2 files in the repo can help to have the pacagges and dependencies;
  befor starting installing and adding the pacagges, check python version install on machinen. here have the file .python-version, you can change it or install
  same on the machine and then activate it
   file pyproject.toml help to install most pacagges, but should have uv installed. after installing uv, just run in the terminal in vs code the line:
   uv sync

   and if prefer using pip, i have the file requirements.txt. just run :
   pip install -r requirements.txt

4. In the project only 2 important files;
   ptbxl_utils.py file: include all dataset configurations but not pythorch classes and dataloader. do not need to do anything about this file
   it is allready called in the main file .

   simple_1dcnn.ipynb file: the important one. It has pythorch class and the model. to run it:
    1. make sure all pacagges are installed by running the first cell.
    2. make sure data_path has access to the dataset on machinen in cell 3.(important with the correct path)
    3. cell 4 is very important. there we can configure the folds, frequecy and choose the data(filtered or full)
       there in cell 4, select whatever folds you want for training, validation or testing, can choose one fold or more.
       use_500Hz help to select 100Hz or 500Hz, only need to change True/False
       use_filtere help to select filterd data( only norm and AFIB) or full data(not finished yet).
       and in same cell 4, all functions in the file ptbxl_utils.py are called.
       after finsishing the configuration, just run the cell to make sure all are correct.

   4. The code is optimized to automatically select either cuda/GPU or cpu, depend on the machine. no need to worry about,
      just check cell 7 and 8 and run them, itwill print if it will be running on GPU or not.


 And in the file an extra file : exploreDataset.ipynb, can help to explore the dataset we use.
