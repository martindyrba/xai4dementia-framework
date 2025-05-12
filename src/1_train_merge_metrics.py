import os
import pandas as pd
import glob

# Define the base path where your x1, x2, ... folders are located
base_path = "./train/2024-10-29_0922_k-fold_earlystrop_MCIcorrect/"

# Define the folder names
folders = ["cv_1", "cv_2", "cv_3", "cv_4", "cv_5"]
subfolders = ["AD-CN", "AD-FTD", "FTD-CN"]

# class repeat
metrics = [
    "TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "Precision",
    "FPR", "FNR", "FDR", "Accuracy", "F1", "AUC",
    "TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "Precision",
    "FPR", "FNR", "FDR", "Accuracy", "F1", "AUC"
]

# Create an empty list to store data
meta_AD_CN = pd.DataFrame()
meta_AD_CN['metrics'] = metrics
meta_AD_CN['class'] = ['AD']*13 + ['CN']*13 

meta_AD_FTD = pd.DataFrame()
meta_AD_FTD['metrics'] = metrics
meta_AD_FTD['class'] = ['AD']*13 + ['FTD']*13 

meta_FTD_CN = pd.DataFrame()
meta_FTD_CN['metrics'] = metrics
meta_FTD_CN['class'] = ['FTD']*13 + ['CN']*13 

meta_temp_1 ={"AD-CN":meta_AD_CN, "AD-FTD":meta_AD_FTD, "FTD-CN":meta_FTD_CN}  

# Loop through each folder and subfolder
for subfolder in subfolders:
    meta_temp={} 
    for folder in folders:
        # Construct the path to the metric.csv file
        csv_path = os.path.join(base_path, folder, subfolder, "metrics.performance.csv")
        
        # Check if the file exists
        if os.path.exists(csv_path):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
        else: 
            continue

        meta_temp[folder] = df['score'] 
    
     # Convert the dictionary to a DataFrame
    score_df = pd.DataFrame.from_dict(meta_temp)
    meta_temp_1[subfolder] =  pd.concat([meta_temp_1[subfolder], score_df], axis=1, ignore_index=False)    
    meta_temp_1[subfolder].to_csv(os.path.join(base_path, '{}_metrics.csv'.format(subfolder)), index=False)
