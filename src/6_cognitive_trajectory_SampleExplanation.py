import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import util_gen as u
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


#Full Data available to us, on which we do the clustering analysis on 
X_full,y_full = u.fetch_expspace(exp_features='w-score', 
                       explanation_space= 'relevance_volumetry_CortThk',         #'relevance_volumetry', 
                       #selected_diagnosis=['AD', 'CN', 'MCI'],                  #'bvFTD'    #only the subjects with given diagnosis where used while finding #N Nearest Neighbours.
                       selected_diagnosis=['AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA'],                 
                       mutualinfo_threshold=True,                                  #only use features which were found imp by MI, to find #N NNs   
                       )
X_full.index = y_full.index



#Data to find neighbours on, i.e., data cohorts with longitudnal infromation
X,y = u.fetch_expspace(exp_features='w-score', 
                       explanation_space= 'relevance_volumetry_CortThk',         #'relevance_volumetry', 
                       #selected_diagnosis=['AD', 'CN', 'MCI'],                  #'bvFTD'    #only the subjects with given diagnosis where used while finding #N Nearest Neighbours.
                       selected_diagnosis=['AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA'],                 
                       mutualinfo_threshold=True,                                  #only use features which were found imp by MI, to find #N NNs   
                       filter_for_longitudnal = True,                            #Only time we use this. It filters all data cohorts except ADNI and DELCODE, as they are the only two cohorts with longitudnal dataset.
                       )            

#https://bib.dbvis.de/uploadedFiles/155.pdf
#Suggests that Manhattan distance metric (p=1) is the most preferable for high dimensional applications
neighbours=10                      #5,10
knn = NearestNeighbors(n_neighbors=neighbours+1,               #finds N nearest neighbour, including the point of query
                       p=2
                       )    
knn.fit(X)     #Note: we only train on the dataset with ADNI/DELCODE samples

query_id = '6849_ADNI' #please give an id suitbale to your cohort here.                     
query_sample = X_full[y_full['fullsid']==query_id] 


_, sim_indices = knn.kneighbors(query_sample)
print(y.loc[y.index[sim_indices[0][1:]]])
similar_samples = y.loc[y.index[sim_indices[0][1:]]]['fullsid'].to_list() 
similar_samples_diag = y.loc[y.index[sim_indices[0][1:]]]['grp_rel'].to_list()           #grp_x
similar_samples_dict = {key.split('_')[0]: value for key, value in zip(similar_samples, similar_samples_diag)}

#Given by MD
data_delcode = pd.read_excel('./6_trajectory/data/Antrag 391_Teipel_DTI-Analysis_20230216_repseudonymisiert_korrigiert.xlsx',
                                    sheet_name=['BL-Daten','FU-Daten'] )    #returns a dict
new_header = data_delcode['BL-Daten'].iloc[0]                   #grab the first row for the header
data_delcode['BL-Daten'] = data_delcode['BL-Daten'][1:]         #take the data less the header row
data_delcode['BL-Daten'].columns = new_header                   #set the header row as the df header
new_header = data_delcode['FU-Daten'].iloc[0] 
data_delcode['FU-Daten'] = data_delcode['FU-Daten'][1:] 
data_delcode['FU-Daten'].columns = new_header 
cols_delcode = ['mmstot','cdrglobal']                           #'cdrtot'


df_adni_mmse_followup = pd.read_csv('./6_trajectory/data/Neuropsychological/MMSE_11Jul2024.csv')
cols_adni_mmse =['MMSCORE']     #keys: PHASE, RID
df_adni_cdr_followup = pd.read_csv('./6_trajectory/data/Neuropsychological/CDR_11Jul2024.csv')
cols_adni_mmse =['CDGLOBAL']     #keys: PHASE, RID



temp = {} 
for sample in similar_samples: 
    if 'ADNI' in sample:
        sample_id = sample.split('_')[0]
        temp_df_mmse = df_adni_mmse_followup[df_adni_mmse_followup['RID']==int(sample_id)][['VISDATE', 'MMSCORE']]   
        temp_df_cdr = df_adni_cdr_followup[df_adni_cdr_followup['RID']==int(sample_id)][['VISDATE', 'CDGLOBAL']]    
        merged_df = pd.merge(temp_df_mmse, temp_df_cdr, on='VISDATE', how='inner')  # Inner join removes unmatched rows
        merged_df = merged_df.dropna() 
        
        temp[sample_id] = { 'MMSE': merged_df['MMSCORE'].to_list(),    #temp_df_mmse
                        'CDR': merged_df['CDGLOBAL'].to_list(), 
                        'Dates': merged_df['VISDATE'].to_list(), 
                      }
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in temp[sample_id]['Dates'] ] 
        baseline_date = dates[0] 
        temp[sample_id]['FU-yrs'] = [(date.year - baseline_date.year) + (date.month - baseline_date.month)/12 + (date.day - baseline_date.day)/365.25 for date in dates]  
        temp[sample_id]['MMSE'] =[int(i) for i in temp[sample_id]['MMSE']]    
        temp[sample_id]['CDR'] =[float(i) for i in temp[sample_id]['CDR']]


    if 'DELCODE' in sample:
        sample_id = sample.split('_')[0]
        temp_df_bl = data_delcode['BL-Daten'][data_delcode['BL-Daten']['Repseudonym']==sample_id][['visdat']+cols_delcode]    
        temp_df_fu = data_delcode['FU-Daten'][data_delcode['FU-Daten']['Repseudonym']==sample_id][['visdat']+cols_delcode]    
        temp[sample_id] = { 'MMSE':  temp_df_bl['mmstot'].to_list() + temp_df_fu['mmstot'].to_list(),
                        'CDR':  temp_df_bl['cdrglobal'].to_list() + temp_df_fu['cdrglobal'].to_list(),    
                        'Dates': temp_df_bl['visdat'].to_list() + [date.strftime('%Y-%m-%d') for date in temp_df_fu['visdat']],  
                      }    
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in temp[sample_id]['Dates'] ] 
        baseline_date = dates[0] 
        temp[sample_id]['FU-yrs'] = [(date.year - baseline_date.year) + (date.month - baseline_date.month)/12 + (date.day - baseline_date.day)/365.25 for date in dates]
        #temp[sample_id]['MMSE'] =[int(i) for i in temp[sample_id]['MMSE']]  
        
        q = [] 
        for i in temp[sample_id]['MMSE']:  
            try:
                q.append(float(i))
            except:
                q.append(np.nan)
        temp[sample_id]['MMSE'] = pd.Series(q).interpolate()     #[float(i) for i in temp[sample_id]['MMSE']] 


        q = [] 
        for i in temp[sample_id]['CDR']:  
            try:
                q.append(float(i))
            except:
                q.append(np.nan)
        temp[sample_id]['CDR'] = pd.Series(q).interpolate()     #[float(i) for i in temp[sample_id]['CDR']]    

sns.set_theme(style='whitegrid')
#plt MMSE
plt.figure()
for sample in temp.keys():
    temp_df = pd.DataFrame({'x': temp[sample]['FU-yrs'], 'y':temp[sample]['MMSE']}  )
    plt.plot(temp_df['x'] , temp_df['y'] , marker='o', label= similar_samples_dict[sample] + ' | ' + sample)
#plt.legend(loc='best')
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Moves legend outside the axes
plt.title('MMSE follow up of most similar patients')
plt.xlabel('Follow-up (Years)')
plt.ylabel('MMSE')
plt.xlim([0, 6])  # we only want to follow people for 6 years/72months
plt.ylim(0,30.5)
plt.minorticks_on()  # Enable minor ticks for extra grid lines
plt.grid(True, which='minor', color='gray', axis='y', linestyle=':', linewidth=0.5)  # Custom minor grid lines
plt.grid(True, which='major', color='black', axis='y', linewidth=0.7)  # Custom minor grid lines
plt.show()


#plt CDR
plt.figure()
for sample in temp.keys():
    temp_df = pd.DataFrame({'x': temp[sample]['FU-yrs'], 'y':temp[sample]['CDR']}  )
    plt.plot(temp_df['x'] , temp_df['y'] , marker='o', label=similar_samples_dict[sample] + ' | ' + sample)
#plt.legend(loc='best')
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Moves legend outside the axes
plt.title('CDR follow up of most similar patients')
plt.xlabel('Follow-up (Years)')
plt.ylabel('CDR Global')
plt.xlim([0, 6])  # we only want to follow people for 6 years/72months
plt.ylim(0,3.1)
plt.show()

print()
