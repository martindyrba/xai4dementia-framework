import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram

#------------------objects------------------------------
palette_hex = [

    "#510011",  #~rosewood dark red

    "#a50026",  #shades of red
    "#d73027",
    "#f46d43",
    "#fdae61",

    "#fbd3a5",  #Yellow-red
    "#faf8e9", #lighter yellow
    "#dee6b8",  #Yellow-Green

    "#b7e1ba",
    "#6ebd91",
    "#3d907d",
    "#2a7371",  #shades of green

    "#1d5061",  #~dark blue-green
      ]
#------------------functions----------------------------
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig = dendrogram(linkage_matrix, **kwargs)
    return fig, linkage_matrix


def full_explanation_space_features(df):
    activation_space, volumetry_space, vol_act_space, ROIs = [],[],[],[]  

    for i in df.columns.to_list(): 
        if '_x' in i:
            if i in ['sid_x','sample_x','Unnamed:0','grp_x','sex1f_x','education_y_x','age_x','MRI_field_strength_x','eTIV_x','MMSE_x','fullsid']:
                pass
            else:
                activation_space.append(i)
                ROIs.append(i.replace('_x',''))

    for i in df.columns.to_list():
        if '_y' in i:
            if i in ['sid_y','sample_y','Unnamed:0','grp_y','sex1f_y','education_y_y','education_y_x','age_y','MRI_field_strength_y','eTIV_y','MMSE_y','fullsid']:
                pass
            else:
                volumetry_space.append(i)

    vol_act_space = activation_space + volumetry_space 
    return activation_space, volumetry_space, vol_act_space, ROIs




def full_explanation_space_features2(df):
    activation_space, volumetry_space, corticalThickness_space, vol_act_cortThk_space, ROIs = [],[],[],[],[]   
   
    #Logically setting the regions not required for relevance to zero. 
    # This includes WM, BrainStem and CSF regions. As the model was only trained on the GM scans.   
    zero_activations = ['Left_Cerebellum_White_Matter_rel', 'Right_Cerebellum_White_Matter_rel', 
                        
                        'Left_Cerebral_White_Matter_rel', 'Right_Cerebral_White_Matter_rel',
                        'White_Matter_rel', 'WM_Hypointensities_rel',
                        'Corpus_Callosum_rel', 'Posterior_rel', 'Mid_Posterior_rel', 'Central_rel', 'Mid_Anterior_rel', 'Anterior_rel',

                        'Brainstem_rel',

                        'Cerebrospinal_Fluid_rel', 'CSF_rel',
                        '3rd-Ventricle_rel', '4th-Ventricle_rel',
                        'Left_Lateral_Ventricle_rel', 'Right_Lateral_Ventricle_rel',
                        'Left_Inf-Lat-Vent_rel', 'Right_Inf-Lat-Vent_rel',
                        'Left_Choroid_Plexus_rel', 'Right_Choroid_Plexus_rel',
                        
                        ] 


    for i in df.columns.to_list():
        if '_rel' in i:
            if i in ['sid_rel','sample_rel','Unnamed:0','grp_rel','sex1f_rel','education_y_rel',
                     'age_rel','MRI_field_strength_rel','eTIV_rel','MMSE_rel','fullsid']:
                pass
            elif i in zero_activations:
                pass
            else:
                activation_space.append(i)
                ROIs.append(i.replace('_rel',''))

    for i in df.columns.to_list():
        if '_vol' in i:
            if i in ['sid_vol','sample_vol','Unnamed:0','grp_vol','sex1f_vol','education_y_vol',
                     'age_vol','MRI_field_strength_vol','eTIV_vol','MMSE_vol','fullsid']:
                pass
            else:
                volumetry_space.append(i)

    for i in df.columns.to_list():
        if '_cortThk' in i:
            if i in ['sid_cortThk','sample_cortThk','Unnamed:0','grp_cortThk','sex1f_cortThk','education_y_cortThk',
                     'age_cortThk','MRI_field_strength_cortThk','eTIV_cortThk','MMSE_cortThk','fullsid']:
                pass
            else:
                corticalThickness_space.append(i)

    return activation_space, volumetry_space, corticalThickness_space, ROIs












def fetch_expspace(exp_features='w-score', 
                   explanation_space='relevance_volumetry', 
                   selected_diagnosis=False,
                   mutualinfo_threshold=True,
                   filter_for_longitudnal=False):
    
    #1.Chose which infomation to encode in the features spaces, i.e., disease specific w-scores or raw activations/volumes
    if exp_features=='w-score':

        #Explanation space created from w-scores of relevance(density), volumetry and cortical thickness signals
        node=1       #node - 0:AD / 1:CN / 2:FTD
        experiment_name = '2024-10-29_0922_k-fold_earlystrop_MCIcorrect'  #'2024-10-23_1105_k-fold_earlystrop_MCIcorrect'
        fold= 'cv_1'  
        rel_method = 'LRP-CMPalpha1beta0' 
        activation_type = 'density_total'
        withDELCODE_data = 'withDelcode'
        file_path = './measures_combined/{}/{}/{}/{}/Rel_Vol_cortThk_w-score-ROIs_{}_{}.csv'.format(experiment_name,fold,rel_method,withDELCODE_data,node,activation_type)
        df_data = pd.read_csv(file_path)    

    elif exp_features=='normal':  #Skipped Code   
        #Explanation space created from actual relevance(density) and vlumetry signals
        df_data = pd.read_csv('Act_Vol_ROIs_cv7_AD+MCIvsCN_1_density_total.csv')    


    #2. Chose how many dimentions to represent the data in, i.e., some specific features or all features 
    if mutualinfo_threshold:
        relvence_features, volumetry_features, cortThk_features, rel_vol_cort_features = read_selected_features(exp_features) 
    else:    #Skipped Code
        relvence_features, volumetry_features, rel_vol_features,_ = full_explanation_space_features(df_data)    #full_explanation_space_features2??
 
 
    #3. Chose which explanation_space to give back to user :[ Relevance, Volumetry, corticalThickness, Relevance_Volumetry_corticalThk ]
    if explanation_space.lower() == 'relevance':
        X = df_data[relvence_features].dropna() 
        y = df_data.loc[X.index][['fullsid','grp_rel']]    
    elif explanation_space.lower() == 'volumetry':
        X = df_data[volumetry_features].dropna()  
        y = df_data.loc[X.index][['fullsid','grp_rel']] 
    elif explanation_space.lower() == 'cortthk':
        X = df_data[cortThk_features].dropna() 
        y = df_data.loc[X.index][['fullsid','grp_rel']] 
    elif explanation_space.lower() == 'relevance_volumetry_cortthk':
        X = df_data[rel_vol_cort_features].dropna() 
        y = df_data.loc[X.index][['fullsid','grp_rel']] 

    #4. Chose the diagnosis groups to be reported to user, i.e., all AD,CN,MCI,svFTD,bvFTD,SMC OR the ones selected below
    if len(selected_diagnosis):
        #y = y[y['grp_x'].isin(['AD', 'CN', 'MCI', 'bvFTD'])]
        X.index = y.index
        y = y[y['grp_rel'].isin(selected_diagnosis)]
        X = X.loc[y.index] 

    #5. Chose if only want to consider datacohorts for which we have the longitudanal scans available
    if filter_for_longitudnal:
        X.index = y.index
        y = y[y['fullsid'].str.contains('DELCODE|ADNI', regex=True, na=False)]   
        X = X.loc[y.index] 

    return X,y







def read_selected_features(features):
    # empty list to read list from a file
    #rel,vol,rel_vol = [],[],[]
    rel,vol,cort,rel_vol_cort = [],[],[],[] 

    # open file and read the content in a list
    if features=='w-score':
        file = r'./3_feature_selection/withDELCODE/selected_ROI_features_thresh0.1.txt'
    elif features=='normal': #SkippedCode
        file=r'./3_feature_selection/raw ralevance + raw vol/selected_ROI_features_thresh0.1.txt'

    with open(file, 'r') as fp:  
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            #x: Rel activations;  y: volumetry
            if '_rel' in x:                        #_x, _y
                rel.append(x)
            if '_vol' in x:
                vol.append(x)
            if '_cortThk' in x:
                cort.append(x)
            rel_vol_cort.append(x)

    return rel, vol, cort, rel_vol_cort
