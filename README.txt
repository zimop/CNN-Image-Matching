To run Autoencoder.ipynb:
run all the cells which do not say #only run this cell if you are making the csv file for submission# at the top of the cell, to generate our logistics
on the training set.
To use the test set and generate the 2000x20 CSV file, run all the cells that do not say #for evaluating training and testing#

The file path that the file works with is from the COMP90086_2023_TLLdataset, an example of a path for the training set is:
"COMP90086_2023_TLLdataset/train/left/aaa.jpg"


Siamese Models:

All of these were run on local Windows system, and are stored in their respectively named folders.
For all files you can run them through command line, but it assumes the csv files, and the train and testing folders are all in the same folder as the python file

Transfer Learnt:
This is the model we found was best for siamese network, using transfer learning.

Triplet Loss:
This one uses triplet loss function

Data Augmentation:
This one uses data augmentation in generating training data.

Hard Negative Mining:
This one does work, but is not computationally feasible.




