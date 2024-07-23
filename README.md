modelcondensed.py is the Python file that compiles the One-Class SVM model for transit detection.

To compile the model:
Open modelcondensed.py in a Python editor. 
Download the KOI_Q16_long dataset file. This houses the PDCSAP flux files that the model uses as a training dataset.
Note that cloning the repository to access the dataset file may not work. To ensure everything runs smoothly, manually downloading the file and referencing it from there is preferable.

Change the directory_path value in the code to wherever the KOI_Q16_long dataset file is stored.

Once the code is run, it should preprocess the data, train and evaluate the model.
