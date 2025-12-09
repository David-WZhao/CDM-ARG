# CDM-ARG

The source code of Conditional Diffusion Model for annotating properties of ARGs(CDM-ARG) in our manuscript.

### Data

The process of collecting data is described in our manuscript and metadata can be accessed upon request.
You need first unzip all compressed files under "data", and put the files in the same directory.

### Environment Deployment

1. Install the required packages: "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple".
2. Install the CUDA version of PyTorch that matches your GPU architecture, and ensure it is compatible with the installed torchvision package.

### How to run the code?

1. Data preprocessing: "arg_v5.fasta" file is the original data set file. The "fasta_process.py" file is used on the original data set file to get the processed dataset. Run "data_divide.py" to produce splitted dataset.

2. Run the prediction model: Put the "data_loader.py", "modules.py", "run.py", "utils.py" and directory "data" in the same directory, and run "python run.py --epoch 10 --K 5" in the command line.

### How to directly test model performance?

1. Data divide: "arg_v5.fasta" file is the original data set file. The "fasta_process.py" file is used on the original data set file to get the processed dataset. Run "data_divide.py" to produce splitted dataset.

2. Run “evaluate_model.py” to skip lengthy model training and directly load our pre-trained model to test its performance.
   We also provide a video for testing model performance locally, which is Test Video.mp4.

# Introduction to Baselines

To comprehensively evaluate the effectiveness of the proposed method, the following three representative methods are selected as baselines for performance comparison:

## BestHit

This method is conducted by comparing the sample sequences with existing ARGs in CARD by applying the BLAST or DIAMOND, and the predicted properties are assigned to samples through applying a similarity cutoff. Note that BestHit can be used only for predicting antibiotic classes and resistance mechanisms of ARGs. For detailed usage of BestHit, please refer to the link:[The Comprehensive Antibiotic Resistance Database (mcmaster.ca)](https://card.mcmaster.ca/analyze/blast)

## DeepARG

This method is a deep learning-based model which is trained by taking the consistency distribution of homologies between sample sequences and all known ARGs as input features. Note that DeepARG can be used only for predicting antibiotic classes of ARGs. For detailed usage of the DeepARG, please refer to the link: https://github.com/gaarangoa/deeparg.

## HMD-ARG

This method extracts features from raw sequences through an end-to-end deep CNN-based framework for predicting properties of ARGs. Note that HMD-ARG is a multi-task model, which can be used for predicting all of three properties of ARGs. For detailed usage of the HMD-ARG, please refer to the link: [http://www.cbrc.kaust.edu.sa/HMDARG](http://www.cbrc.kaust.edu.sa/HMDARG/).
