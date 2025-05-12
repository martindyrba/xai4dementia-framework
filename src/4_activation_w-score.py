import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import text_exp_util2 as u
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import os

#ROI vol file
df_vol = pd.read_csv('./volume/FastSurfer_Volumes_combined3.csv', index_col=False) 
onto_file = 'tree-structure-brain-anatomy-FastSurfer-v3.rdf'
LRModels_dict = 'ROI_model_dict.pkl'
df_vol, df_vol_w, root, parents_list, list_ROIs, children_dict_id_to_name  = u.w_score_wrapper_vol(LRModels_dict, df_vol, onto_file)

#Activation ROI file
node=0  #node - 0/1/2
experiment_name = '2024-10-29_0922_k-fold_earlystrop_MCIcorrect'  #'2024-10-23_1105_k-fold_earlystrop_MCIcorrect'
fold= 'cv_1'  
rel_method = 'LRP-CMPalpha1beta0'  #'IG','LRP'
fname = './act_seg/{}/{}/{}/SegmentedActivations_node{}_processed.xlsx'.format(experiment_name,fold,rel_method,node)
if os.path.isfile(fname): 
  activation_type = 'density_total'
  xls = pd.ExcelFile(fname)    
  df_density_total = pd.read_excel(xls, 'density_total', index_col=False)                        #activationfiles - act_total, act_sum_pos, act_sum_neg, density_total
  df_act = df_density_total

else: #find the segmented activations for all regions  #TODO: redo the file strcuture format
  xls = pd.ExcelFile('./act_seg/SegmentedActivations_node{}.xlsx'.format(node))    #node - 0/1
  df_act_sum_pos = pd.read_excel(xls, 'act_sum_pos', index_col=False)                        #activationfiles - act_total, act_sum_pos, act_sum_neg
  df_act_sum_neg = pd.read_excel(xls, 'act_sum_neg', index_col=False)                        
  df_act_total = pd.read_excel(xls, 'act_total', index_col=False)                        
  df_density_sum_pos = pd.read_excel(xls, 'density_sum_pos', index_col=False)                      
  df_density_sum_neg = pd.read_excel(xls, 'density_sum_neg', index_col=False)                        
  df_density_total = pd.read_excel(xls, 'density_total', index_col=False)                       
  df_no_vox = pd.read_excel(xls, 'no_voxles', index_col=False)

  df_act_sum_pos = u.activation_df_preprocess(df_act_sum_pos,df_vol_w)
  df_act_sum_neg = u.activation_df_preprocess(df_act_sum_neg,df_vol_w)
  df_act_total = u.activation_df_preprocess(df_act_total,df_vol_w)
  df_density_sum_pos = u.activation_df_preprocess(df_density_sum_pos,df_vol_w)                   
  df_density_sum_neg = u.activation_df_preprocess(df_density_sum_neg,df_vol_w)
  df_density_total = u.activation_df_preprocess(df_density_total,df_vol_w)
  df_no_vox = u.activation_df_preprocess(df_no_vox,df_vol_w)

  df_act_sum_pos,_,df_density_sum_pos    = u.find_parent_activaton(df_act_sum_pos, df_no_vox, parents_list, root,df_density_sum_pos)
  df_act_sum_neg,_,df_density_sum_neg    = u.find_parent_activaton(df_act_sum_neg, df_no_vox, parents_list, root,df_density_sum_neg)
  df_act_total,df_no_vox,df_density_total = u.find_parent_activaton(df_act_total, df_no_vox, parents_list, root,df_density_total)


  df_act_sum_pos.rename(columns=children_dict_id_to_name, inplace=True)
  df_act_sum_neg.rename(columns=children_dict_id_to_name, inplace=True)
  df_act_total.rename(columns=children_dict_id_to_name, inplace=True)
  df_density_sum_pos.rename(columns=children_dict_id_to_name, inplace=True)
  df_density_sum_neg.rename(columns=children_dict_id_to_name, inplace=True)
  df_density_total.rename(columns=children_dict_id_to_name, inplace=True)
  df_no_vox.rename(columns=children_dict_id_to_name, inplace=True)

  with pd.ExcelWriter('./act_seg/SegmentedActivations_node{}_processed.xlsx'.format(node)) as writer:
      df_act_sum_pos.to_excel(writer,sheet_name='act_sum_pos')
      df_act_sum_neg.to_excel(writer,sheet_name='act_sum_neg')
      df_act_total.to_excel(writer,sheet_name='act_total')
      df_density_sum_pos.to_excel(writer,sheet_name='density_sum_pos')
      df_density_sum_neg.to_excel(writer,sheet_name='density_sum_neg')
      df_density_total.to_excel(writer,sheet_name='density_total')
      df_no_vox.to_excel(writer,sheet_name='no_voxles')




#------------------------create w-score of ACTIVATIONs-------------------------------------  

#a) Train a LR model for CN samples; save as dict  
df_act_CN = df_act[df_act['grp']=='CN'] 

def train_lr(ROI,df_CN):

  X = df_CN[['sex1f','age','MRI_field_strength','eTIV']]
  y = df_CN[str(ROI)]   #Volume/Activation column for a specific ROI

  #2. Training the models
  model_LR = LinearRegression().fit(X.values, y.values)
  residual = y.values - model_LR.predict(X.values)
  std_residual =np.std(residual)
  return model_LR, std_residual

ROI_model_dict_act = {}
#Loop over all the ROI; train LR model. 
for ROI in list_ROIs:
  subroi_dict = {}
  model, std_residual = train_lr(str(ROI),df_act_CN)
  subroi_dict['model_coef'] = np.squeeze(model.coef_)
  subroi_dict['model_intercept'] = model.intercept_
  #temporarily saving the residuals, to be added up at higher parent abstractions
  subroi_dict['std_residual'] = std_residual
  ROI_model_dict_act[str(ROI)] = subroi_dict

#Save LR model
with open( './act_seg/{}/{}/{}/ROI_model_dict_act_node{}_density.pkl'.format(experiment_name,fold,rel_method,node), 'wb') as f:
  pickle.dump(ROI_model_dict_act, f)


#b) find the w-score 
df_w_act = df_act.copy(deep=True) #a new dataframe for storing w-scores
    
#Find the w-score for the ROIs
for ROI in list_ROIs:
    ROI = str(ROI)
    df_w_act[ROI] = u.calc_w_batch(col_measured_vol= df_act[ROI].values,
                                    col_covariate_values = df_act[['sex1f','age','MRI_field_strength','eTIV']].values,
                                    ROI=ROI,
                                    ROI_model_dict=ROI_model_dict_act
                                    )
    
#df_w_act.to_csv('activation_w-score_node{}_density.csv'.format(node), index=False)
df_w_act.drop(df_w_act.columns[0],axis=1,inplace=True)





#--------------------------------------
#read cortical thickness measures

df_cort_thickness_w = pd.read_csv('./thickness_surfacearea/ThickAvg_w-scores.csv')

# Add a suffix (_cortThk) to all columns except the excluded fullsid
df_cort_thickness_w = df_cort_thickness_w.rename(columns={col: f"{col}_cortThk" for col in df_cort_thickness_w.columns if col != 'fullsid'})




##------------------------------ merge and save--------------

merged_df1 = pd.merge(df_w_act, df_vol_w, on='fullsid', how='outer', suffixes=('_rel', '_vol'))
final_merged_df = pd.merge(merged_df1, df_cort_thickness_w, on='fullsid', how='outer')


#df_plot = pd.merge(df_w_act, df_vol_w, on='fullsid')      #df_act -> x ; df_vol -> y'
#Save this merged dataframe for later usage


#df_plot.to_csv('Act_Vol_w-score-ROIs_{}_{}_{}.csv'.format('cv7_AD+MCIvsCN',node,activation_type),index=False)

df_filtered = final_merged_df[~final_merged_df['fullsid'].str.contains('DELCODE|DESCRIBE', regex=True, na=False)]   
os.makedirs( './measures_combined/{}/{}/{}'.format(experiment_name,fold,rel_method)
              , exist_ok=True)
df_filtered.to_csv('./measures_combined/{}/{}/{}/Rel_Vol_cortThk_w-score-ROIs_{}_{}.csv'.format(experiment_name,fold,rel_method,node,activation_type), index=False)
#final_merged_df.to_csv('./measures_combined/{}/{}/{}/Rel_Vol_cortThk_w-score-ROIs_{}_{}.csv'.format(experiment_name,fold,rel_method,node,activation_type), index=False)


print()
