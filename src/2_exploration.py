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
df_vol = pd.read_csv('./volume/FastSurfer_Volumes_combined3_parent.csv', index_col=False)    #pd.read_csv('./volume/new_FS_vol_combined.csv', index_col=False)
with open('./volume/list_ROIs.pkl', 'rb') as f:
  list_ROIs =  pickle.load(f)


#Activation ROI file
node=0  #node - 0:AD/1:CN/2:BV
experiment_name = '2024-10-29_0922_k-fold_earlystrop_MCIcorrect' #'2024-10-23_1105_k-fold_earlystrop_MCIcorrect'
fold= 'cv_1'  
rel_method = 'LRP-CMPalpha1beta0'  #'IG', LRP-CMPalpha1beta0, IG_alibi_, DeepTaylor_Bounded
fname = './act_seg/{}/{}/{}/SegmentedActivations_node{}_processed.xlsx'.format(experiment_name,fold,rel_method,node)
if os.path.isfile(fname): 

  xls = pd.ExcelFile(fname)    

  activation_type = 'activation_total'
  df_density_total = pd.read_excel(xls, 'act_total', index_col=False)                            #activationfiles - act_total, act_sum_pos, act_sum_neg

  #activation_type = 'density_total'
  #df_density_total = pd.read_excel(xls, 'density_total', index_col=False)                         #activationfiles - act_total, act_sum_pos, act_sum_neg

  df_act = df_density_total

else: #find the segmented activations for all regions
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




#------------------------------------Exploration----------------------------------------


df_plot = pd.merge(df_act, df_vol, on='fullsid')      #df_act -> x ; df_vol -> y
#Save this merged dataframe for later usage

#Scatter Plots
for ROI in list_ROIs:
  df_temp = df_plot[['grp_x', '{}_x'.format(ROI), '{}_y'.format(ROI)]] 
  #df_temp = df_temp[ (df_temp[ 'grp_x'] == 'CN')|  (df_temp[ 'grp_x'] == 'AD')]    #Only plotting Ad-vs-CN atm.
  #df_temp = df_temp[ (df_temp[ 'grp_x'] == 'CN') |  (df_temp[ 'grp_x'] == 'bvFTD')]    
  plt.figure()
  ax = sns.scatterplot(data=df_temp, x='{}_x'.format(ROI), y='{}_y'.format(ROI), hue='grp_x')
  ax.set(xlabel='{}: {}'.format(ROI,activation_type), ylabel='{}: volume'.format(ROI))
  # save the plot as PNG file
  os.makedirs('./scatterplot/{}/{}/{}/node{}_{}'.format(experiment_name,fold,rel_method,node,activation_type), exist_ok=True)
  plt.savefig("./scatterplot/{}/{}/{}/node{}_{}/ROI:{}.png".format(experiment_name,fold,rel_method,node,activation_type,ROI))
  df_temp=None

# Corealtion: Between activations and volumes
corr = {} 
for ROI in list_ROIs:
  df_temp = df_plot[['{}_x'.format(ROI), '{}_y'.format(ROI)]].dropna()   #We drop the rows/ROIs where activations were absent, and have nan values. 
  corr[str(ROI)] = pearsonr(df_temp['{}_x'.format(ROI)], df_temp['{}_y'.format(ROI)])
df_corr = pd.DataFrame.from_dict(corr, orient='index')
df_corr = df_corr.reset_index()
df_corr.to_csv('./act_seg/{}/{}/{}/pearson_r_node{}_{}.csv'.format(experiment_name,fold,rel_method,node,activation_type), index=False)

