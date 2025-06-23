# CDM-ARG
The source code of Conditional Diffusion Model for annotating properties of ARGs(CDM-ARG) in our manuscript.

### Requirement
- python 3.8
- torch == 1.12.1
- scikit-learn == 1.2.2
- numpy == 1.24.2
- pandas == 1.3.4

### Data
The process of collecting data is described in our manuscript and metadata can be accessed upon request.
You need first unzip all compressed files under "data", and put the files in the same directory.


### How to run the code?
1. Data preprocessing: "arg_v5.fasta" file is the original data set file, "fasta_process.ipynb" file is used on the original data set file to get the processed dataset.
Run "data_divide.py" to produce splitted dataset.

2. Run the prediction model: Put the "data_loader.py", "modules.py", "run.py", "utils.py" and directory "data" in the same directory, and run "python run.py --device "cuda:0" --batch_size 32 --epoch 10" in the command line.

# Introduction to Baselines
To comprehensively evaluate the effectiveness of the proposed method, the following three representative methods are selected as baselines for performance comparison:

## BestHit
This method is conducted by comparing the sample sequences with existing ARGs in CARD by applying the BLAST or DIAMOND, and the predicted properties are assigned to samples through applying a similarity cutoff. Note that BestHit can be used only for predicting antibiotic classes and resistance mechanisms of ARGs. For detailed usage of BestHit, please refer to the link:[The Comprehensive Antibiotic Resistance Database (mcmaster.ca)](https://card.mcmaster.ca/analyze/blast)

## DeepARG
This method is a deep learning-based model which is trained by taking the consistency distribution of homologies between sample sequences and all known ARGs as input features. Note that DeepARG can be used only for predicting antibiotic classes of ARGs. For detailed usage of the DeepARG, please refer to the link: https://github.com/gaarangoa/deeparg.

## HMD-ARG
This method extracts features from raw sequences through an end-to-end deep CNN-based framework for predicting properties of ARGs. Note that HMD-ARG is a multi-task model, which can be used for predicting all of three properties of ARGs. For detailed usage of the HMD-ARG, please refer to the link: [http://www.cbrc.kaust.edu.sa/HMDARG](http://www.cbrc.kaust.edu.sa/HMDARG/).
