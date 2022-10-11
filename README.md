# SleepXAI: An Explainable Deep Learning approach for Multi-class Sleep Stage Identification #

## SleepXAI: An Explainable Deep Learning approach for Multi-class Sleep Stage Identification ##

### Requirements ###
Python 2.7 <br />
tensorflow/tensorflow-gpu<br />
numpy<br />
scipy<br />
matplotlib<br />
scikit-learn<br />
matplotlib<br />
pandas<br />
mne<br />

### To download SC subset of subjects from the Sleep_EDF (2013) dataset: ### 
cd data_2013  <br />
chmod +x download_physionet.sh  <br />
./download_physionet.sh <br />


### Once the data is downloaded, follow the steps to run the SleepXAI model ###  
Simply run "run.py" to generate four different models for EEG Fpz-Cz and Pz-Oz signals with CNN-CNN and CNN-CRF models. Confusion matrices are also generated automatically with 20-fold cross-validation. <br /> 


### Once the models are created, generate hypnogram and grad-cam visualization ### 
run "cnn_hypnogram.py", "crf_hypnogram.py" and "grad_cam.py" <br />


