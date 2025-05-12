import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.feature_selection import mutual_info_classif             #Mutual Information: MI
import operator
import matplotlib.pyplot as plt
import numpy as np  
import pickle
import os
import util_gen as u

#Explanation space created from actual relevance(density) and vlumetry signals
#df_exp = pd.read_csv('Act_Vol_ROIs_cv7_AD+MCIvsCN_1_density_total.csv')

#Explanation space created from w-scores of relevance(density) and vlumetry signals
df_exp = pd.read_csv('./measures_combined/2024-10-29_0922_k-fold_earlystrop_MCIcorrect/cv_1/LRP-CMPalpha1beta0/Rel_Vol_cortThk_w-score-ROIs_1_density_total.csv')
#df_exp = pd.read_csv('Act_Vol_w-score-ROIs_cv7_AD+MCIvsCN_1_density_total.csv')
#x: Rel activations;  y: volumetry

#activation_space, volumetry_space, vol_act_space, ROIs = u.full_explanation_space_features(df_exp)
activation_space, volumetry_space, corticalThickness_space, ROIs =  u.full_explanation_space_features2(df_exp)

#Volumetry
df_temp = df_exp[volumetry_space + ['grp_rel']].dropna()
res1 = dict(zip(volumetry_space,
               mutual_info_classif(df_temp[volumetry_space]  , df_temp['grp_rel'])   #  grp_x-->grp_rel
               ))

res1_sorted = dict( sorted(res1.items(), key=operator.itemgetter(1), reverse=True))
#print(res1)


#Activation
df_temp = df_exp[activation_space + ['grp_rel']].dropna()
#Removes limited row (n=6) for which model did not gave attention to some regions
res2 = dict(zip(activation_space,
               mutual_info_classif(df_temp[activation_space], df_temp['grp_rel'])     
               ))

res2_sorted = dict( sorted(res2.items(), key=operator.itemgetter(1), reverse=True))
#print(res2)

#Cortical Thickness
df_temp = df_exp[corticalThickness_space + ['grp_rel']].dropna(axis=1, how='all')   #remove all the subcortical regions without cortical measures
corticalThickness_space = df_temp.columns.to_list()[:-1] 
df_temp = df_temp.dropna()
res3 = dict(zip(corticalThickness_space,
               mutual_info_classif(df_temp[corticalThickness_space], df_temp['grp_rel'])  
               ))

res3_sorted = dict( sorted(res3.items(), key=operator.itemgetter(1), reverse=True))
#print(res3)




#Volumetry+Activation+Cortical_Thickness space
vol_act_cortThk_space = activation_space + volumetry_space + corticalThickness_space
df_temp = df_exp[vol_act_cortThk_space + ['grp_rel']].dropna()
#Removes limited row (n=6) for which model did not gave attention to some regions
res4 = dict(zip(vol_act_cortThk_space,
               mutual_info_classif(df_temp[vol_act_cortThk_space], df_temp['grp_rel'])
              ))

res4_sorted = dict( sorted(res4.items(), key=operator.itemgetter(1), reverse=True))


#-------------------Bar charts-------------------------------

#plt.xticks(rotation='vertical')
#plt.bar(list(res1_sorted.keys()), list(res1_sorted.values()))   #res1,2,3


#--------------------GroupedbarChart---------------------------


def MI_min(feature, MI_dict):
    try: 
        value = MI_dict[feature]
        return value
    except: 
        return 0       #Zero is minimum possible mutual information value, logically represnting independent features. 
                       #This should pop us null in computations for feature selection, wo breaking the code



#ROIs_t = tuple(ROIs)                                  #t: tuple
#ROIs_t = tuple( i.replace('_x','') for i in res2_sorted.keys())       #RelevanceSorted
ROIs_t = tuple( i.replace('_vol','') for i in res1_sorted.keys())        #VolumeSorted

vol_t, rel_t, cort_t = (),(),()

for i in ROIs_t:
    vol_t = vol_t +    (MI_min(i+'_vol',res1),)  
    rel_t = rel_t +    (MI_min(i+'_rel',res2),) 
    cort_t = cort_t +  (MI_min(i+'_cortThk',res3),) 
bar_dict = {'Volumetry': vol_t, 'Relevance': rel_t, 'Cortical Thickness': cort_t} 



x = np.arange(len(ROIs_t))  # the label locations
width = 0.2  # the width of the bars   0.2 , 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in bar_dict.items():                                  
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
   # ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mutual Information')
ax.set_title('Region Of Interests (ROIs)')
ax.set_xticks(x + width, ROIs_t)
ax.legend(loc='upper right', ncols=1)
plt.xticks(rotation='vertical')

plt.show()


#----------------------------------------------------------------------------------------------

threshold = 0.1                      
selected_ROI_features = []             #The list of features (volumetry or relevance) for which the MI exceeds the threshold
for i in res4.keys():                  #res3. A dictionary of vol+rel features, and their MI with the disease stages
    if res4[i] > threshold:
        selected_ROI_features.append(i)  



# Create a DataFrame for easier handling
df = pd.DataFrame(bar_dict)
df.index = ROIs_t

# Check which features are above the threshold
above_threshold = df > threshold

# Assign each feature a "combination" based on which sets exceed the threshold
combinations = []
for idx, row in above_threshold.iterrows():
    combo = tuple(row)  # Tuple of True/False values for each set
    combinations.append(combo)

# Convert combinations into a DataFrame for categorization
df['Combination'] = combinations

# Categorize into 7 groups
categories = {
    (True, False, False): "Only Volumetry",
    (False, True, False): "Only Relevance",
    (False, False, True): "Only CortThk",
    (True, True, False):  "Volumetry and Relevance",
    (True, False, True):  "Volumetry and CortThk",
    (False, True, True):  "Relevance and CortThk",
    (True, True, True):   "All measurements agree",
    (False, False, False): "No measurement relevant",
}

df['Category'] = df['Combination'].map(categories)
os.makedirs('./3_feature_selection', exist_ok=True)
df.to_csv ('./3_feature_selection/FeatureCombinations_MI_thresh{}.csv'.format(threshold))

with open('./3_feature_selection/selected_ROI_features_thresh{}.txt'.format(threshold), 'w') as f:
    for line in selected_ROI_features:
        f.write(f"{line}\n")



print()
