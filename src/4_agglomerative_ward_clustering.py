import seaborn as sns
import util_gen as u
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import v_measure_score, adjusted_mutual_info_score, homogeneity_score, completeness_score 
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, silhouette_score, davies_bouldin_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


#------------- STEP 1: COMPARE different feature set against each other, under different settings ------------------
#Mostly, used for evaluating the quality of clusters. 
#Makes it really relative, as the number of clusters dictates the value we get for each of the metric. 
selected_diagnosis = [ 'AD', 'CN']     


#ideal #n clusters
k = 2 #len(selected_diagnosis)

for standarised in [False]:
    for features in ['w-score']:                                                    #'w-score'/normal
        for space in [ 'relevance', 'volumetry', 'cortthk' ,'relevance_volumetry_CortThk']:                                       #[Relevance, Volumetry, Relevance_Volumetry]   
            for feature_selection in [True]:                                        #Use MI featureSelection True/False
                X,y = u.fetch_expspace(exp_features= features, 
                    explanation_space= space,           #[Relevance, Volumetry, Relevance_Volumetry] 
                    selected_diagnosis= selected_diagnosis,               #Should be: [ 'AD', 'CN']
                    mutualinfo_threshold= feature_selection)



                #-----------Clusterig eval metrics-----------------
                model = AgglomerativeClustering(linkage='ward',              #linkage{‘ward’, ‘complete’, ‘average’, ‘single’}
                                                n_clusters=k)
                if standarised: #SkippedCode
                    scaler = StandardScaler().fit(X)
                    X = scaler.transform(X)
                clustering = model.fit(X)
                labels_pred = clustering.labels_
                labels_true = y.grp_rel     #y.grp_x

                v = v_measure_score(labels_true, labels_pred)
                mInf = adjusted_mutual_info_score(labels_true, labels_pred)
                hom = homogeneity_score(labels_true, labels_pred)
                comp = completeness_score(labels_true, labels_pred)
                rand = adjusted_rand_score(labels_true, labels_pred)
                fms = fowlkes_mallows_score(labels_true, labels_pred)

                sil =  silhouette_score(X, labels_pred)
                dbs = davies_bouldin_score(X, labels_pred)

                metric = { 'features':features,
                        'space': space,
                        'feature_selection':feature_selection,
                        
                        'v_measure_score':v,
                        'adjusted_mutual_info_score' : mInf,
                        'homogeneity_score': hom,
                        'completeness_score': comp,
                        'adjusted_rand_score': rand,
                        'fowlkes_mallows_score': fms,

                        'silhouette_score':sil,
                        'davies_bouldin_score':dbs,

                        'standarised': standarised,
                        } 

                try:  #with each seting of the pipeline, we read and write into this file, one line at a time
                    df = pd.read_csv('./4_clustering/metrics_{}.csv'.format("".join(selected_diagnosis)))
                except:
                    df = pd.DataFrame(columns=metric.keys())

                df = pd.concat([df, pd.DataFrame([metric] )], ignore_index=True)
                df.to_csv('./4_clustering/metrics_{}.csv'.format("".join(selected_diagnosis)),index=False)










##-----------------------------  STEP2: WARD, clusterMAP  ---------------------------------
import copy

#Use all while creating the dendogram
#selected_diagnosis = [ 'AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA', 'SMC']     #SMC comes from delcode
selected_diagnosis = [ 'AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA']     
#selected_diagnosis = [ 'AD', 'CN' ,'MCI']
#selected_diagnosis = [ 'AD', 'CN', 'bvFTD']     
X,y = u.fetch_expspace(exp_features= 'w-score', 
                    explanation_space= 'relevance_volumetry_CortThk',           #[Relevance, Volumetry, Relevance_Volumetry] 
                    selected_diagnosis= selected_diagnosis,               #Should be: [ 'AD', 'CN']
                    mutualinfo_threshold= True)
diagnosis_color = y.grp_rel.map({ 'CN':'blue', 'MCI':'pink', 'AD':'red', 'SMC':'orange', 'bvFTD':'green', 'svFTD':'green', 'PNFA':'green' })

X_temp = copy.deepcopy(X)
X_temp.columns = (X_temp.columns
                    .str.replace('_', ' ', regex=True)
                    .str.replace('rel', 'relevance', regex=True)
                    .str.replace('vol', 'volume', regex=True)
                    .str.replace('cortThk', 'cortical thickenss', regex=True)

             )


sns.clustermap( X_temp, method='ward',
                row_colors=diagnosis_color,
                cmap=sns.color_palette(u.palette_hex), center=0, vmin=-5, vmax=5,
            )





##--------------Step3:TO produce pie charts realted to a clustermap-----------
import matplotlib.pyplot as plt
import numpy as np
model = AgglomerativeClustering(linkage='ward',             
                                 n_clusters=2)               #want to seprate the converter/stable people
clustering = model.fit(X)
labels_pred = clustering.labels_
labels_true = y.grp_rel.replace('bvFTD','FTD').replace('svFTD','FTD') 
cluster_no = 0                      #0,1,2,3...
labels_cluster = []  
for i in range(len(labels_pred)):
    if labels_pred[i]==cluster_no:
        labels_cluster.append(labels_true.to_list()[i])
q,r = np.unique(labels_cluster, return_counts=True)
fig, ax = plt.subplots()
ax.pie(r, labels=q, autopct='%1.1f%%',
       colors=[ 'red','blue', 'green','pink'])



print()


# ---------Step4:  4-CV inter rater agreement (05-03-2025) ---------
from statsmodels.stats import inter_rater as irr
from sklearn.model_selection import KFold, StratifiedKFold     #ShuffleSplit I think, since there is no real train-and-test, i.e., test set is just thrown away, we could use the ShuffleSplit here.
from scipy.spatial.distance import cdist
import numpy as np

selected_diagnosis = [ 'AD', 'CN' ,'MCI', 'bvFTD', 'svFTD', 'PNFA'] 
X,y = u.fetch_expspace(exp_features= 'w-score', 
                    explanation_space= 'relevance_volumetry_CortThk',           #[Relevance, Volumetry, Relevance_Volumetry] 
                    selected_diagnosis= selected_diagnosis,               #Should be: [ 'AD', 'CN']
                    mutualinfo_threshold= True)
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
labels_true = y.grp_rel.replace('bvFTD','FTD').replace('svFTD','FTD').replace('PFNA','FTD') 
df_rater_assignements = y.copy()

model = AgglomerativeClustering(linkage='ward',              #linkage{‘ward’, ‘complete’, ‘average’, ‘single’}
                                 n_clusters=2)               #want to seprate the converter/stable people


#kf = KFold(n_splits=4)
skf = StratifiedKFold(n_splits=4)
#ss = ShuffleSplit(train_size=0.9, test_size=0.1, n_splits = 4)

for i, (train_index,test_index) in enumerate(skf.split(X,labels_true)):
    train_set = X.loc[train_index]
    test_set = X.loc[test_index]

    train_y = y.loc[train_index]  #Ground Truth
    train_y = train_y.grp_rel.replace('bvFTD','FTD').replace('svFTD','FTD').replace('PFNA','FTD') 

    clustering = model.fit(train_set)
    # Get cluster labels for training data
    labels_pred = clustering.labels_       # 0 or 1, the meaning of a stable to stable or converter here could change between folds.

    #Since AgglomerativeClustering lacks .predict(), 
    # we assign new points (in the test set) to the closest cluster centroid (or a similar representative point).
    
    #Book-keeping
    df_train = pd.DataFrame(train_set)
    df_train['ground_truth'] = train_y.to_list()             
    df_train['Cluster_assigned'] = labels_pred             


    #Check and restore, if the meaning of cluster0/1 is flipped wrt. stable/converters
    c_cn = df_train['ground_truth']=='CN'
    c_0 = df_train['Cluster_assigned']==0    
    c_1 = df_train['Cluster_assigned']==1
    CN_count_c0 = len(df_train[c_cn & c_0])
    CN_count_c1 = len(df_train[c_cn & c_1])
    if CN_count_c0 > CN_count_c1:    #0:stable / 1:converter. AS we wanted.
        pass
    else:                            #1:stable / 0:converter. FLIP.
        df_train['Cluster_assigned'] = df_train['Cluster_assigned'].replace({0: 1, 1: 0})



    # Compute centroids of the clusters
    # Calculate cluster centroids
    centroids = df_train.groupby('Cluster_assigned').mean(numeric_only =True).values
    # Compute distance of each new point to each cluster centroid
    distances = cdist(test_set, centroids, metric='euclidean')
    # Assign each new point to the nearest cluster
    new_labels = np.argmin(distances, axis=1)


    df_rater_assignements.loc[train_index, 'fold-{}'.format(i)] = df_train['Cluster_assigned']
    df_rater_assignements.loc[test_index, 'fold-{}'.format(i)] = new_labels

arr = df_rater_assignements[['fold-0','fold-1','fold-2','fold-3']] 
agg = irr.aggregate_raters(arr)        #aggregate all the raters. Creates (n_rows, n_cat) - Contains counts of raters that assigned a category (cluster label) to individuals.
kappa = irr.fleiss_kappa(agg[0], method='fleiss')   #Fleiss’s kappa statistic for inter rater agreement. 
print(kappa)










