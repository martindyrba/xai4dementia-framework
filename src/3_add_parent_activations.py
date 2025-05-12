
import pandas as pd
import text_exp_util2 as u


#ROI vol file
df_vol = pd.read_csv('./volume/FastSurfer_Volumes_combined3.csv', index_col=False)
onto_file = 'tree-structure-brain-anatomy-FastSurfer-v3.rdf'
LRModels_dict = './volume/ROI_model_dict_vol.pkl'          
df_vol, df_vol_w, root, parents_list, list_ROIs, children_dict_id_to_name  = u.w_score_wrapper_vol(LRModels_dict, df_vol, onto_file)


#Activation ROI file
experiment_name = '2024-10-29_0922_k-fold_earlystrop_MCIcorrect'    
fold= '_1'                               #The fold from which we are using the 3D-CNN model.
node= 2                                  #node - 0-AD/1-CN/2-FD
rel_method= 'LRP-CMPalpha1beta0'                                    #IG_alibi, LRP, LRP-CMPalpha1beta0, IG_alibi_,DeepTaylor_Bounded
xls = pd.ExcelFile('./act_seg/{}/cv{}/{}/SegmentationActivation_node{}.xlsx'.format(experiment_name,fold,rel_method,node))               

df_act_sum_pos = pd.read_excel(xls, 'act_sum_pos', index_col=False)                        #activationfiles - act_total, act_sum_pos, act_sum_neg
df_act_sum_neg = pd.read_excel(xls, 'act_sum_neg', index_col=False)                        
df_act_total = pd.read_excel(xls, 'act_total', index_col=False)                        
df_density_sum_pos = pd.read_excel(xls, 'density_sum_pos', index_col=False)                      
df_density_sum_neg = pd.read_excel(xls, 'density_sum_neg', index_col=False)                        
df_density_toal = pd.read_excel(xls, 'density_total', index_col=False)                       
df_no_vox = pd.read_excel(xls, 'no_voxles', index_col=False)

df_act_sum_pos = u.activation_df_preprocess(df_act_sum_pos,df_vol_w)
df_act_sum_neg = u.activation_df_preprocess(df_act_sum_neg,df_vol_w)
df_act_total = u.activation_df_preprocess(df_act_total,df_vol_w)
df_density_sum_pos = u.activation_df_preprocess(df_density_sum_pos,df_vol_w)                   
df_density_sum_neg = u.activation_df_preprocess(df_density_sum_neg,df_vol_w)
df_density_toal = u.activation_df_preprocess(df_density_toal,df_vol_w)
df_no_vox = u.activation_df_preprocess(df_no_vox,df_vol_w)

df_act_sum_pos,_,df_density_sum_pos    = u.find_parent_activaton(df_act_sum_pos, df_no_vox, parents_list, root,df_density_sum_pos)
df_act_sum_neg,_,df_density_sum_neg    = u.find_parent_activaton(df_act_sum_neg, df_no_vox, parents_list, root,df_density_sum_neg)
df_act_total,df_no_vox,df_density_toal = u.find_parent_activaton(df_act_total, df_no_vox, parents_list, root,df_density_toal)


df_act_sum_pos.rename(columns=children_dict_id_to_name, inplace=True)
df_act_sum_neg.rename(columns=children_dict_id_to_name, inplace=True)
df_act_total.rename(columns=children_dict_id_to_name, inplace=True)
df_density_sum_pos.rename(columns=children_dict_id_to_name, inplace=True)
df_density_sum_neg.rename(columns=children_dict_id_to_name, inplace=True)
df_density_toal.rename(columns=children_dict_id_to_name, inplace=True)
df_no_vox.rename(columns=children_dict_id_to_name, inplace=True)

with pd.ExcelWriter('./act_seg/{}/cv{}/{}/SegmentedActivations_node{}_processed.xlsx'.format(experiment_name,fold,rel_method,node)) as writer:
    df_act_sum_pos.to_excel(writer,sheet_name='act_sum_pos', index=False)
    df_act_sum_neg.to_excel(writer,sheet_name='act_sum_neg', index=False)
    df_act_total.to_excel(writer,sheet_name='act_total', index=False)
    df_density_sum_pos.to_excel(writer,sheet_name='density_sum_pos', index=False)
    df_density_sum_neg.to_excel(writer,sheet_name='density_sum_neg', index=False)
    df_density_toal.to_excel(writer,sheet_name='density_total', index=False)
    df_no_vox.to_excel(writer,sheet_name='no_voxles', index=False)
