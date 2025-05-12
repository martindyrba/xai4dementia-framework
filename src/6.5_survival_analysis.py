from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
from lifelines.plotting import loglogs_plot
import util_gen as u
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
 

#--------------------Clusering to find subcluster memebers--------------------
#selected_diagnosis = ['AD', 'CN', 'MCI']                  #'bvFTD'    #only the subjects with given diagnosis where used while finding #N Nearest Neighbours. 
selected_diagnosis = [ 'AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA']     
X,y = u.fetch_expspace(exp_features='w-score', 
                       explanation_space= 'relevance_volumetry_CortThk',         #'relevance_volumetry', 
                       selected_diagnosis= selected_diagnosis,
                       mutualinfo_threshold=True,                                  #only use features which were found imp by MI, to find #N NNs   
                       ) 
k = 2 #2 #len(selected_diagnosis)=3

model = AgglomerativeClustering(linkage='ward',           
                                n_clusters=k)
clustering = model.fit(X)
labels_pred = clustering.labels_
labels_true = y.grp_rel                                       #grp_x


#--------- Convert Longitudnal data (From ADNI and Delcode) into Survival Framework -------------
#--DELCODE--
data_delcode = pd.read_excel('./6_trajectory/data/Antrag 391_Teipel_DTI-Analysis_20230216_repseudonymisiert_korrigiert.xlsx',
                                    sheet_name=['BL-Daten','FU-Daten'] )    #returns a dict
new_header = data_delcode['BL-Daten'].iloc[0]                   #grab the first row for the header
data_delcode['BL-Daten'] = data_delcode['BL-Daten'][1:]         #take the data less the header row
data_delcode['BL-Daten'].columns = new_header                   #set the header row as the df header
data_delcode['BL-Daten'] = data_delcode['BL-Daten'][['Repseudonym', 'visdat','mmstot','cdrglobal']] 
data_delcode['BL-Daten'] = data_delcode['BL-Daten'].dropna()

new_header = data_delcode['FU-Daten'].iloc[0] 
data_delcode['FU-Daten'] = data_delcode['FU-Daten'][1:] 
data_delcode['FU-Daten'].columns = new_header 
data_delcode['FU-Daten'] = data_delcode['FU-Daten'][['Repseudonym', 'visdat','mmstot','cdrglobal']]  
data_delcode['FU-Daten'] = data_delcode['FU-Daten'].dropna()

delcode_age_sex = pd.read_csv('./6_trajectory/data/xxx_DELCODE_t1_info_complete.csv',
                                    usecols=['RID', 'Sex1F', 'Age'])


cols_delcode = ['mmstot','cdrglobal']                           #'cdrtot'



#--ADNI--
df_adni_age_check = pd.read_excel('./6_trajectory/data/Kopie von extended_sample_CDR.xlsx',
                                  sheet_name=['extended_sample'] )
df_adni_age_check = df_adni_age_check['extended_sample'][['RID','T1_Scan_Date']]          
df_adni_age_sex = pd.read_excel('./6_trajectory/data/ADNI combined.xlsx',
                                  sheet_name=['sample'])
df_adni_age_sex = df_adni_age_sex['sample'][['RID','Sex (1=female)', 'Age at scan']]          


df_adni_mmse_followup = pd.read_csv('./6_trajectory/data/Neuropsychological/MMSE_11Jul2024.csv', 
                                    usecols=['RID', 'VISDATE', 'MMSCORE'])
df_adni_mmse_followup = df_adni_mmse_followup.dropna()
df_adni_cdr_followup = pd.read_csv('./6_trajectory/data/Neuropsychological/CDR_11Jul2024.csv', 
                                   usecols=['RID', 'VISDATE', 'CDGLOBAL'])
df_adni_cdr_followup = df_adni_cdr_followup.dropna()

cols_adni = ['MMSCORE', 'CDGLOBAL'] 


temp = {} 
for sample in y['fullsid'].to_list():                              #similar_samples: 
    if 'ADNI' in sample:
        sample_id = int(sample.split('_')[0])
        temp_df_mmse = df_adni_mmse_followup[df_adni_mmse_followup['RID']==sample_id]
        temp_df_mmse['VISDATE'] = pd.to_datetime(temp_df_mmse['VISDATE'])
        #temp_df_mmse['VIS-Yr'] =    pd.DatetimeIndex(temp_df_mmse['VISDATE']).year 
 
        temp_df_cdr = df_adni_cdr_followup[df_adni_cdr_followup['RID']==sample_id]
        temp_df_cdr['VISDATE'] = pd.to_datetime(temp_df_cdr['VISDATE'])
        #temp_df_cdr['VIS-Yr'] =    pd.DatetimeIndex(temp_df_cdr['VISDATE']).year 

        
        # For each row in df1, find the nearest date in df2 and calculate the difference
        merged_rows = []
        for i, row in temp_df_mmse.iterrows():
            mmse_date = row['VISDATE']
            # Find the nearest date in the cdr df for the current date1
            nearest_cdr_date = temp_df_cdr.loc[(temp_df_cdr['VISDATE'] - mmse_date).abs().idxmin()]['VISDATE']
            # Calculate the difference between date1 (mmse date) and the nearest (cdr) date
            time_diff = abs(nearest_cdr_date - mmse_date)
            # If the difference is less than 2 weeks, append the result to merged_rows
            if time_diff <= timedelta(weeks=2):
                merged_rows.append({'RID': sample_id,
                                    'MMSCORE': temp_df_mmse[temp_df_mmse['VISDATE'].dt.date == mmse_date.date()]['MMSCORE'].iloc[0],
                                    'CDGLOBAL': temp_df_cdr[temp_df_cdr['VISDATE'].dt.date == nearest_cdr_date.date()]['CDGLOBAL'].iloc[0],
                                    'VISDATE': mmse_date, 
                                    })
            # Convert the merged rows into a DataFrame
            merged_df = pd.DataFrame(merged_rows)


        #merged_df = temp_df_mmse.merge(temp_df_cdr, how = 'outer', on = ['VIS-Yr'])
        merged_df[cols_adni] = merged_df[cols_adni].apply(pd.to_numeric, errors='coerce')   #datacleaning
        merged_df = merged_df.dropna()    
        merged_df = merged_df[(merged_df['MMSCORE'] >= 0) & (merged_df['MMSCORE'] <= 30)]
        merged_df = merged_df[(merged_df['CDGLOBAL'] >= 0) & (merged_df['CDGLOBAL'] <= 3)]
        
        #logic for removing observarions before the 1st T1 scan used
        if sample_id in df_adni_age_check['RID'].to_list():
            date_T1 = df_adni_age_check[sample_id == df_adni_age_check['RID']]['T1_Scan_Date']  
           # merged_df['VISDATE'] = pd.to_datetime(merged_df['VISDATE'])
            reference_date = pd.to_datetime(date_T1.iloc[0])
            # Calculate 3 months before the reference date
            three_months_before = reference_date - pd.DateOffset(months=3)
            # Filter the DataFrame to keep rows where 'datetime_column' is within the last 3 months before reference_date
            merged_df = merged_df[merged_df['VISDATE'] >= three_months_before]

            if len(merged_df) == 0: #i.e. no data poins are present after correcting for first T1-scan as baseline
                continue
        
        temp[sample] = { 'MMSE': merged_df['MMSCORE'].astype(int).to_list() , 
                        'CDR': merged_df['CDGLOBAL'].astype(float).to_list() , 
                        'Dates': merged_df['VISDATE'].to_list(), 
                        'age' :  float(df_adni_age_sex[sample_id == df_adni_age_sex['RID']]['Age at scan']),
                        'sex_1f':int(df_adni_age_sex[sample_id == df_adni_age_sex['RID']]['Sex (1=female)']) 
                      }
        #dates = [datetime.strptime(date, "%Y-%m-%d") for date in temp[sample]['Dates'] ] 
        dates = temp[sample]['Dates']
        baseline_date = dates[0] 
        temp[sample]['FU-months'] = [(date.year - baseline_date.year)*12 + (date.month - baseline_date.month) + (date.day - baseline_date.day)/30.437 for date in dates]  


            

    if 'DELCODE' in sample:
        sample_id = sample.split('_')[0]
        temp_df_bl = data_delcode['BL-Daten'][data_delcode['BL-Daten']['Repseudonym']==sample_id][['visdat']+cols_delcode]    
        temp_df_fu = data_delcode['FU-Daten'][data_delcode['FU-Daten']['Repseudonym']==sample_id][['visdat']+cols_delcode]
        #temp_df_fu['visdat'] =  [date.strftime('%Y-%m-%d') for date in temp_df_fu['visdat']]

        merged_df = pd.concat([temp_df_bl,temp_df_fu], ignore_index=True)
        merged_df['visdat'] = pd.to_datetime(merged_df['visdat'])
        merged_df[cols_delcode] = merged_df[cols_delcode].apply(pd.to_numeric, errors='coerce')    #datacleaning
        merged_df = merged_df.dropna()
        
        merged_df = merged_df[(merged_df['mmstot'] >= 0) & (merged_df['mmstot'] <= 30)]
        merged_df = merged_df[(merged_df['cdrglobal'] >= 0) & (merged_df['cdrglobal'] <= 3)]
        
        temp[sample] = { 'MMSE':  merged_df['mmstot'].astype(int).to_list(),
                        'CDR':  merged_df['cdrglobal'].astype(float).to_list(),    
                        'Dates': merged_df['visdat'].to_list(),  
                        'age' :  float(delcode_age_sex[sample_id == delcode_age_sex['RID']]['Age']),
                        'sex_1f':int(delcode_age_sex[sample_id == delcode_age_sex['RID']]['Sex1F']) 
                      }    
        #dates = [datetime.strptime(date, "%Y-%m-%d") for date in temp[sample]['Dates'] ]
        dates = temp[sample]['Dates']
        baseline_date = dates[0] 
        temp[sample]['FU-months'] = [(date.year - baseline_date.year)*12 + (date.month - baseline_date.month) + (date.day - baseline_date.day)/30.437 for date in dates]



'''
#-------------------------- mean cluster cognitivie trajectory plots (MMSE/CDR) -------------------------------

def plt_mean_std_line(x_values, y_values, xlabel, ylabel, ylim, title, cluster_no, flag='mmse', gdr_truth=[]):
    sns.set_theme(style='whitegrid')    
    # Define a common x-axis: Use the union of all x-values or a defined range
    common_x = np.linspace(0, max([max(x) for x in x_values]), 60)

    # Interpolate each line to the common x-axis
    interpolated_y = []
    n = len(x_values)
    for x, y in zip(x_values, y_values):
        if len(x)==1:
            interp_func = interp1d(x, y, kind='nearest', fill_value="extrapolate")      
        elif len(x)>=2:
            interp_func = interp1d(x, y, kind='slinear', fill_value="extrapolate")
        
        #elif len(x)>=3:             ###THESE HIGHER ORDER SPLINE FUNCTIONS DO NOT REALLY MATCH OUR DATA WELL. diverges alot
        #    interp_func = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
        #elif len(x)>=4:
        #    interp_func = interp1d(x, y, kind='cubic', fill_value="extrapolate")      
        
        
        #interp_func = interp1d(x, y, kind='nearest', fill_value="extrapolate")              #zero, nearest
        interpolated_y.append(interp_func(common_x))

    interpolated_y = np.array(interpolated_y)

    # Calculate the mean and standard deviation across the interpolated lines
    y_mean = np.mean(interpolated_y, axis=0)
    y_std = np.std(interpolated_y, axis=0)

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot the mean line
    plt.plot(common_x, y_mean, label='Mean', color='black')

    # Fill between mean ± standard deviation
    plt.fill_between(common_x, y_mean - y_std, y_mean + y_std, color='lightskyblue', alpha=0.2, label='Standard Deviation')

    # plot the original lines for reference
    if len(gdr_truth):         #i.e., we want to color the lines by thier baseline diagnosis
        for (x,y, diag) in zip(x_values,y_values, gdr_truth):
            if diag=='CN':
                cl = 'blue'
            elif diag=='MCI':
                cl='pink'
            elif diag=='AD':
                cl='red'            
            plt.plot(x, y, color=cl, alpha=0.4, marker='.' )        
    else:
        for (x,y) in zip(x_values,y_values):
            plt.plot(x, y, color='gray', alpha=0.2, marker='.' )


    # Labels and title
    plt.title(title.format(cluster_no) + '| #patients:{}'.format(n))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if flag.lower()=='mmse':
        plt.minorticks_on()  # Enable minor ticks for extra grid lines
        plt.grid(True, which='minor', color='gray', axis='y', linestyle=':', linewidth=0.5)  # Custom minor grid lines
        plt.grid(True, which='major', color='black', axis='y', linewidth=0.7)  # Custom minor grid lines
    plt.legend()
    plt.savefig('./6.5_cluster_trajectory/{}{}_spline01'.format(flag,cluster_no))


#Average MMSE/CDR Plots
for cluster_no in range(k):
    x_values,y_mmse,y_cdr = [],[],[] 
    #optional
    gdr_truth = []   
    for sample in y[labels_pred == cluster_no]['fullsid']:
        if (sample.split('_')[-1] == 'ADNI') or (sample.split('_')[-1] == 'DELCODE'):
            x_values.append(temp[str(sample.split('_')[0])]['FU-months'])   
            y_mmse.append(temp[str(sample.split('_')[0])]['MMSE'])   
            y_cdr.append(temp[str(sample.split('_')[0])]['CDR'])  
            #baseline_assesement of the MRI scan
            gdr_truth.append( y[y['fullsid'] == sample]['grp_x'].to_list()[0]   )

    plt_mean_std_line(x_values, y_mmse, 'Followup (in months)', 'MMSE', (0,30.5),'Mean MMSE for Cluster{}', cluster_no, 'mmse')
    plt_mean_std_line(x_values, y_cdr, 'Followup (in months)', 'CDR', (0,3.1) ,'Mean CDR for Cluster{}', cluster_no, 'cdr')

'''



#------------------ for linear modeling (with repeated measures) for cluster averages in R -------------
df_ = pd.DataFrame(columns=['RID','MMSE', 'CDR', 'FUMonths', 'cluster', 'age', 'sex_1f', 'baseline_diag'] )

for cluster_no in range(k):
    for sample in y[labels_pred == cluster_no]['fullsid']:
        if (('ADNI' in sample ) or ('DELCODE' in sample )) and (sample in temp.keys()):
            for observation_index in range(len(temp[sample]['MMSE'])) :
                row = [ sample, 
                        temp[sample]['MMSE'][observation_index], 
                        temp[sample]['CDR'][observation_index], 
                        temp[sample]['FU-months'][observation_index],
                        cluster_no,
                        temp[sample]['age'],
                        temp[sample]['sex_1f'],
                        y[y['fullsid'] == sample]['grp_rel'].iloc[0]        #grp_x  
                    ]
                df_.loc[len(df_.index)] = row  
df_.to_csv('R_repeated_measures_{}clusters.csv'.format(k), index=False)



#--------------------------- survival plots per cluster ------------------------------------


for cluster_no in range(k):
    df = pd.DataFrame(columns=['T', 'E', 'base_CDR'] )               #A new survival plot for each cluster
    for sample in y[labels_pred == cluster_no]['fullsid']:
        if (('ADNI' in sample ) or ('DELCODE' in sample )) and (sample in temp.keys()):

            #1st condition: we want samples with atleast one followup,i.e., 2 observations
            if not len(temp[sample]['MMSE']) > 1: 
                continue
            
            #A temp dataframe for processing
            df_temp = pd.DataFrame(columns=['MMSE', 'CDR', 'Dates']) 
            df_temp['MMSE'] = temp[sample]['MMSE']  
            df_temp['CDR'] = temp[sample]['CDR']  
            df_temp['Dates'] = pd.DatetimeIndex(temp[sample]['Dates'])  
            df_temp.sort_values(by='Dates',inplace=True)
            
            #2nd condition: we do not consider people who get better cognitively for survival analysis
            if not df_temp['CDR'].is_monotonic_increasing:
                continue           
            
            #Logic for convering logitudnal data to survival dataframe
            last_cdr = None
            base_cdr = None
            time_index = None
            for index, row in df_temp.iterrows():
                if last_cdr is None:            #intialize the counter
                    last_cdr = row['CDR'] 
                    base_cdr = row['CDR']
                    time_index = row['Dates']
                    continue  

                if last_cdr >= 1:      #We do not care about progression beyond the first AD assement
                    break
                
                if row['CDR'] > last_cdr:
                    #process CDR change
                    months_lapsed = (row['Dates'] - time_index).days/30.437
                    df.loc[len(df.index)] = [months_lapsed, 1, base_cdr]     #Event 1 (death): Conversion CN-to-MCI or MCI-to-AD
                    #re-initialize
                    last_cdr = row['CDR'] 
                    time_index = row['Dates']
            if (row['Dates']>time_index) and (last_cdr<1):
                months_lapsed = (row['Dates'] - time_index).days/30.437
                df.loc[len(df.index)] = [months_lapsed, 0, base_cdr]        #Event 0 (survival): No conversion of the diagnosis

    #plt.figure()
    #sns.set_theme(style='whitegrid')    

    if cluster_no==0:
        c = 'red' #'green'
        grp_name = 'converters'
    elif cluster_no==1:
        c = 'blue' #'orange'
        grp_name = 'stable'

    else:
        c = 'blue'

    kmf = KaplanMeierFitter(label='{}'.format(grp_name))
    kmf.fit(df['T'], df['E'])

    kmf.plot(show_censors=True, color=c)

    #add_at_risk_counts(kmf, xticks = [0,12, 24, 36, 48, 60, 72, 84, 96, 108] )

    plt.xlabel('Follow up (in months)')
    plt.ylabel('Probability of staying dementia free (%)')
    plt.ylim((-0.1,1.1))
    plt.xlim((0,72))
    plt.title('Kaplan Meier Survival Curve| Cluster{}'.format(grp_name))    #cluster_no
#    plt.minorticks_on()  # Enable minor ticks for extra grid lines
 #   plt.grid(True, which='minor', color='gray', axis='y', linestyle=':', linewidth=0.5)  # Custom minor grid lines
    plt.grid(True, which='major', color='black', axis='y', linewidth=0.5)  # Custom minor grid lines
    plt.grid(True, which='major', color='black', axis='x', linewidth=0.5)  # Custom minor grid lines
    plt.tight_layout()

    #plt.savefig('./6.5_cluster_trajectory/KM_cluster{}__'.format(cluster_no)) 
plt.show()
print()




