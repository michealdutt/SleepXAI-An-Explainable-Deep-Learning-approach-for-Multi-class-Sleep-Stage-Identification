import os

list_run = ["run_eeg_fpz_cz_crf.py", "run_eeg_fpz_cz_cnn.py", "run_eeg_pz_oz_cnn.py", "run_eeg_pz_oz_crf.py"] 

for file in list_run:
    print(file)
    os.system("python " + str(file))
