import os, logging
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
logging.getLogger('innvestigate').disabled = True

import segmentation_util as tu
from keras  import models
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   #To only select the CPU.
import tensorflow as tf
tf.compat.v1.disable_eager_execution() ## required by innvestigate to be able to analyze the network graph
import innvestigate


node = 0         # Binary:[0,1];  multi-class:[0,1,2...], where  AD:0, CN:1, FTD:2           

x_range_from = 0
x_range_to = 193
y_range_from = 0
y_range_to = 229
z_range_from = 0
z_range_to = 193
minmaxscaling = True


ip_shape = (193,229,193,1)
op_shape = 3    
fold_no = 1
cv_fold= 'cv_{}'.format(fold_no)   #01, cv_1
exp_name = '2024-10-29_0922_k-fold_earlystrop_MCIcorrect'   
model_name = 'ADMCIvsCNvsFTD_cv0{}.h5'.format(fold_no)
model_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Text_Exp/train/{}/{}/{}'.format(exp_name, cv_fold, model_name)   
model = tu.DenseNet(ip_shape,op_shape)
model = models.load_model(model_path)
# replace last layer's softmax activation function by linear
model_wo_softmax = innvestigate.model_wo_softmax(model)
# create analyzer
analyzer = innvestigate.create_analyzer(tu.Relevance_Method[0][0], 
                                        model_wo_softmax, 
                                        **tu.Relevance_Method[0][1])


data_path = '/ssd1/dyrba/xxx_t1linear_N4cor_WarpedSegmentationPosteriors2_allinone/'

#dict_distribution = {} 
cols = ['Sid','Cohort'] + list(tu.labels_lookup.keys()) 
df_act_sum_pos = pd.DataFrame(columns=cols)
df_act_sum_neg = pd.DataFrame(columns=cols)
df_act_sum_total = pd.DataFrame(columns=cols)           #total relevance: neg+pos
df_density_sum_pos = pd.DataFrame(columns=cols)
df_density_sum_neg = pd.DataFrame(columns=cols)
df_density_sum_total = pd.DataFrame(columns=cols)
df_no_voxels = pd.DataFrame(columns=cols)

print('Processing #{} samples'.format(len(os.listdir(data_path))))
print('Proceesing for CNNs node# {}'.format(node))

error_cases =[] 
#Loop over each sample in the data directory
for i,sample_file in enumerate(os.listdir(data_path)):
    (cohort_code, sid) = tu.find_sampleID(sample_file)
    print('index:{}, {} sample:{}'.format(i,cohort_code,sid))
    
    try:
        #Load the FastSurfer's (already, affine transformed from native space to t1-linear space) segmentation file.
        segment_vol = tu.read_nifti_data( tu.find_segemetnation_file(sample_file), 
                                        x_range_from, x_range_to, 
                                        y_range_from, y_range_to, 
                                        z_range_from, z_range_to, 
                                        minmaxscaling=False)
        segment_vol = segment_vol.astype(int) #converts float codes to int
    except:
        error_cases.append('{}_{} : Segmentation Error'.format(cohort_code,sid))
        continue

    try:
        #Creating (innvestigate's) LRP activations  
        sample_nifti =  tu.read_nifti_data(os.path.join(data_path,sample_file), 0, 193, 0, 229, 0, 193, minmaxscaling=True) 
        activation_vol = tu.get_sample_activation(analyzer, sample_nifti, node)  
    except:
        error_cases.append('{}_{} : Activation Map Error'.format(cohort_code,sid))
        continue

#   dict_roi={}
    row_act_sum_pos = [sid,cohort_code] 
    row_act_sum_neg = [sid,cohort_code] 
    row_act_sum_total = [sid,cohort_code] 
    row_density_sum_pos = [sid,cohort_code] 
    row_density_sum_neg = [sid,cohort_code]
    row_density_sum_total = [sid,cohort_code]
    row_no_voxels = [sid,cohort_code]

    for roi in tu.labels_lookup.keys():
        seg_mask = np.where(segment_vol== tu.labels_lookup[roi], True, False)
        temp = {}   
        no_voxels = np.sum(seg_mask)          #Number of voxels      
        row_no_voxels.append(no_voxels)        
        seg_act = activation_vol*seg_mask      #Masking

        act_sum_pos = tu.sign_sum(seg_act, 'pos')
        row_act_sum_pos.append(act_sum_pos)
        act_sum_neg = tu.sign_sum(seg_act, 'neg')
        row_act_sum_neg.append(act_sum_neg)
        act_sum_total = act_sum_pos+act_sum_neg
        row_act_sum_total.append(act_sum_total)

        row_density_sum_pos.append(act_sum_pos/no_voxels)
        row_density_sum_neg.append(act_sum_neg/no_voxels)
        row_density_sum_total.append(act_sum_total/no_voxels)

    df_act_sum_pos.loc[len(df_act_sum_pos)] = row_act_sum_pos
    df_act_sum_neg.loc[len(df_act_sum_neg)] = row_act_sum_neg
    df_act_sum_total.loc[len(df_act_sum_total)] = row_act_sum_total
    
    df_density_sum_pos.loc[len(df_density_sum_pos)] = row_density_sum_pos
    df_density_sum_neg.loc[len(df_density_sum_neg)] = row_density_sum_neg
    df_density_sum_total.loc[len(df_density_sum_total)] = row_density_sum_total

    df_no_voxels.loc[len(df_no_voxels)] = row_no_voxels
    

save_path = './act_seg/{}/{}/{}'.format(exp_name, cv_fold, tu.Relevance_Method[0][2] )
tu.dir_check(save_path, non_empty_dir_ok=True)

with pd.ExcelWriter(os.path.join(save_path, 'SegmentationActivation_node{}.xlsx'.format(node))) as writer:
    df_act_sum_pos.to_excel(writer,sheet_name='act_sum_pos')
    df_act_sum_neg.to_excel(writer,sheet_name='act_sum_neg')
    df_act_sum_total.to_excel(writer,sheet_name='act_total')
    df_density_sum_pos.to_excel(writer,sheet_name='density_sum_pos')
    df_density_sum_neg.to_excel(writer,sheet_name='density_sum_neg')
    df_density_sum_total.to_excel(writer,sheet_name='density_total')
    df_no_voxels.to_excel(writer,sheet_name='no_voxles')


#Save error cases
with open( os.path.join(save_path, 'Error_cases_node{}.txt'.format(node)),'w') as f:
    for line in error_cases:
        f.write(f"{line}\n")
