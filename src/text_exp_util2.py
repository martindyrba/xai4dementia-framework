import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import pickle
import os

from owlready2 import set_render_func
from owlready2 import get_ontology
from owlready2 import default_world

from anytree import AnyNode, Node, RenderTree
from anytree import search
#from anytree.search import find
from anytree import PostOrderIter

from operator import add

def calc_w_batch(col_measured_vol=0.0, col_covariate_values=None, ROI=None, ROI_model_dict=None):
  '''
  A function to calculate w-scores for any ROI, for a given sample.

  col_measured_vol:   the measured/ground_truth ROI volumes for a given region,
                    actually, could be used for the activations as well. :)
  col_covariate_values: (array) list of covariate information assocaited with a given sample,
                      used for predicting expected ROI vol levels
  ROI: (str) name of the ROI,
  ROI_model_dict: (dict) a object containing all the learned LR models with
                  thier coefficients, intercept. Also contain the
                  std. dev. of the control sample's residuals.

  '''
  #Empty model instance
  model = LinearRegression()
  #setting trained weights (model's coef and intercept)
  model.coef_ = ROI_model_dict[ROI]['model_coef']
  model.intercept_ = ROI_model_dict[ROI]['model_intercept']
  std_dev = ROI_model_dict[ROI]['std_residual']

  sample_x = np.array(col_covariate_values)

  w = (col_measured_vol - model.predict(sample_x)) / std_dev

  return w








#For pretty printing of the entitiy names
def render_using_label(entity):
    return entity.label.first() or entity.name
set_render_func(render_using_label)


def parents_pair_func():
  #Find the pairs of parent-child relationships

  # query all parent classes with root class Brain_Region in ontology
  # request only direct children, not grand(grand)children pairs
  parents_pairs = list(default_world.sparql("""
  SELECT ?toplevel ?leaf
      WHERE {
          ?leaf rdfs:subClassOf ?toplevel;
                rdfs:subClassOf* tree-structure-brain-anatomy-FastSurfer-v3:Brain_Region.
  }
  """)) # prefixes are automatically defined, tree-structure-brain-anatomy-FastSurfer: is the currently loaded ontology

  #print('parents_pairs')
  #print(len(parents_pairs), parents_pairs)
  return parents_pairs

def add_children_to_tree(root,onto):
  parents = list()
  class_map = dict()
  parents_pairs = parents_pair_func()

  # first: add children and parents to the map
  for pp in parents_pairs:
    if pp[0] not in class_map:
      class_map[pp[0]] = [pp[1]] # add new list with child as first entry
    else:
      class_map[pp[0]].append(pp[1]) # add child class to existing list in map
    if pp[0] not in parents: parents.append(pp[0])

  # second: recursively add children and parents to the tree
  def add_children(p, vals):
    for v in vals:
      pp = AnyNode(parent=p, ontoClass=v) # create child nodes for each value
      add_children(pp, class_map.get(v, [])) # recursively process children

  add_children(root, class_map[onto.Brain_Region])
 # print('parents')
 # print(len(parents), parents)

  return root, parents_pairs, parents

def child_leaf_names(onto):
  #Find the names of all the chilren/leaf nodes
  children_name, children_id = [], []
  for region in onto.Brain_Region.descendants(include_self=False):
      with onto:
        if len(region.descendants(include_self=False))==0:  # leaf node
          assert region.ID, f"region ID not set for {region}"
          #children.append(str(region.name))
          children_name.append(str(region.name))
          children_id.append(region.ID[0])

  return children_name, children_id



def find_parent_volume(df, parents, root):
  '''
  Additively finds true/ground_truth volumetric measurements for all parent ROIs

  df: dataframe with children ROI volumes. The ROI names are still 'id' format.
  parents: str list, a list of all the parent ROI names.
  root: anytree root node, represnting the whole ontology as a tree.
  '''
  for parent in parents:
    vol=0
    #finds all the leaf children (or child-ROI) for any give parent node
    relevant_leaf_children = search.find(root, lambda node: node.ontoClass.name == str(parent)).leaves
    for ele in relevant_leaf_children:
      vol += df[str(ele.ontoClass.ID[0])]

    df[str(parent)] = vol
  return df

def w_score_wrapper_vol(LRModels_dict, df, finename):
    #Get the OWL/Protege ontology which encodes neuroanatomical structral information
    onto = get_ontology(finename).load()

    # Create an empty tree,
    # to later represent the parent-child relationships amongst ROIs and easily traverse amongst them
    root = Node("root", ontoClass=onto.Brain_Region)
    root, _, parents = add_children_to_tree(root, onto)
    children_name, children_id = child_leaf_names(onto)
    children_dict_id_to_name = dict(zip(children_id, children_name))

    list_ROIs = children_name + parents  #A list of all the ROI, in a human readable format

    df = find_parent_volume(df, parents, root)   #Finding the volume to parent nodes
    #Renaming children ROI columns in the dataframe (from id -> readable name)
    df.rename(columns=children_dict_id_to_name, inplace=True)

    #Read a dictionary contianing the model coeficents and intercepts for all ROIs.
    #This dict also contains std deviation of the control sample's residuals,
    #for easy calculation of the w-score
    with open(LRModels_dict, 'rb') as f:
        ROI_model_dict = pickle.load(f)
        
    #a new dataframe for storing w-scores
    df_w = df.copy(deep=True)
        
    #Find the w-score for the ROIs
    for ROI in list_ROIs:
        ROI = str(ROI)
        df_w[ROI] = calc_w_batch(col_measured_vol= df[ROI].values,
                                        col_covariate_values = df[['sex1f','age','MRI_field_strength','eTIV']].values,
                                        ROI=ROI,
                                        ROI_model_dict=ROI_model_dict
                                        )

    return df, df_w, root, parents, list_ROIs, children_dict_id_to_name


def misc_processing(df_act, df_no_vox, df_act_density):
     #misc processing of select ROI
  df_act['Brain-Stem'] = df_act['Left-Brain-Stem'] +  df_act['Right-Brain-Stem']
  try:
    df_no_vox['Brain-Stem'] = df_no_vox['Left-Brain-Stem'] +  df_no_vox['Right-Brain-Stem']
  except:    pass  
  df_act_density['Brain-Stem'] =  df_act['Brain-Stem'] / df_no_vox['Brain-Stem']


  df_act['CSF'] = df_act['Left-CSF'] +  df_act['Right-CSF']
  try: 
    df_no_vox['CSF'] = df_no_vox['Left-CSF'] +  df_no_vox['Right-CSF']
  except:    pass  
  df_act_density['CSF'] = df_act['CSF'] /  df_no_vox['CSF']


  df_act['3rd-Ventricle'] = df_act['Left-3rd-Ventricle'] +  df_act['Right-3rd-Ventricle']
  try:
    df_no_vox['3rd-Ventricle'] = df_no_vox['Left-3rd-Ventricle'] +  df_no_vox['Right-3rd-Ventricle']
  except:    pass  
  df_act_density['3rd-Ventricle'] = df_act['3rd-Ventricle'] /  df_no_vox['3rd-Ventricle']

  df_act['4th-Ventricle'] = df_act['Left-4th-Ventricle'] +  df_act['Right-4th-Ventricle']
  try:
    df_no_vox['4th-Ventricle'] = df_no_vox['Left-4th-Ventricle'] +  df_no_vox['Right-4th-Ventricle']
  except:    pass  
  df_act_density['4th-Ventricle'] = df_act['4th-Ventricle'] /  df_no_vox['4th-Ventricle']

  df_act.drop(columns=['Left-Brain-Stem', 'Right-Brain-Stem', 
                       'Left-CSF', 'Right-CSF', 
                       'Left-3rd-Ventricle','Right-3rd-Ventricle', 
                       'Left-4th-Ventricle','Right-4th-Ventricle'], 
                       inplace=True)
  try:
     df_no_vox.drop(columns=['Left-Brain-Stem', 'Right-Brain-Stem', 
                       'Left-CSF', 'Right-CSF', 
                       'Left-3rd-Ventricle','Right-3rd-Ventricle', 
                       'Left-4th-Ventricle','Right-4th-Ventricle'], 
                       inplace=True)
  except: pass
  df_act_density.drop(columns=['Left-Brain-Stem', 'Right-Brain-Stem', 
                       'Left-CSF', 'Right-CSF', 
                       'Left-3rd-Ventricle','Right-3rd-Ventricle', 
                       'Left-4th-Ventricle','Right-4th-Ventricle'], 
                       inplace=True)
  return df_act, df_no_vox, df_act_density


def find_parent_activaton(df_act,df_no_vox, parents, root, df_act_density):
  '''
  Additively finds true/ground_truth volumetric measurements for all parent ROIs

  df_act: dataframe with children ROI's activations. The ROI names are still 'id' format.
          The activations either come from sheets ['act_sum_pos','act_sum_neg','act_total']
          DO NOT use density metrics, as this analysis is not meant for density metrics.  
  df_no_vox: dataframe with children ROI's number of voxels. The ROI names are still 'id' format.
  parents: str list, a list of all the parent ROI names.
  root: anytree root node, represnting the whole ontology as a tree.
  '''
  (df_act,df_no_vox, df_act_density) = misc_processing(df_act,df_no_vox, df_act_density)

  for parent in parents:
    activation=0
    voxels=0
    #finds all the leaf children (or child-ROI) for any give parent node
    relevant_leaf_children = search.find(root, lambda node: node.ontoClass.name == str(parent)).leaves
    for ele in relevant_leaf_children:
      activation += df_act[str(ele.ontoClass.ID[0])]
      voxels     += df_no_vox[str(ele.ontoClass.ID[0])]

    #Save activation(sum and density), and total voxels for each parent ROI
    df_act[str(parent)] = activation  
    df_no_vox[str(parent)] = voxels
    df_act_density[str(parent)] = activation / voxels  

  return df_act,df_no_vox,df_act_density


      
def sid_support(row):     #activation_df

    if 'adni' in row['sample'].lower():
        if row['sample'] == 'ADNI3':
          return str(row['sid']) + '_' + 'ADNI'
        else:
          return str(row['sid']).split('_')[1] + '_' + 'ADNI' 

    elif 'nifd' in row['sample'].lower():
       return str(row['sid']).split('_')[1] + '_' + 'NIFD' 

    else:
        return str(row['sid']) + '_' + row['sample']


def sid_support2(row):    #volume_df
    if 'adni' in row['sample'].lower():
        return str(row['sid']).split('_')[1] + '_' + 'ADNI'
    else:
        return row['fullsid']


def activation_df_preprocess(df_act, df_vol):
#  df_act['Brain-Stem'] = df_act['Left-Brain-Stem'] +  df_act['Right-Brain-Stem']
#  df_act['CSF'] = df_act['Left-CSF'] +  df_act['Right-CSF']
#  df_act['3rd-Ventricle'] = df_act['Left-3rd-Ventricle'] +  df_act['Right-3rd-Ventricle']
#  df_act['4th-Ventricle'] = df_act['Left-4th-Ventricle'] +  df_act['Right-4th-Ventricle']
#  df_act.drop(columns=['Left-Brain-Stem', 'Right-Brain-Stem', 
#                       'Left-CSF', 'Right-CSF', 
#                       'Left-3rd-Ventricle','Right-3rd-Ventricle', 
#                       'Left-4th-Ventricle','Right-4th-Ventricle'], 
#                       inplace=True)

#  df_act.rename( columns={'Sid':'sid', 'Cohort':'sample', 'C<ohort':'sample'} , inplace=True)
  df_act.rename( columns={'Sid':'sid', 'Cohort':'sample'} , inplace=True)
  
  #For ADNI2 samples, remove the diagnosis from the sid 
  #FROM the activation datafile
  #for i in df_act.index:
  #  if df_act['sample'][i] in ['ADNI2', 'NIFD']:
  #      df_act['sid'][i] =  df_act['sid'][i].split('_')[1]

#  df_act['sample'].replace('ADNI2/GO','ADNI', inplace=True)   
#  df_act['sample'].replace('ADNI3','ADNI', inplace=True)     
#  df_act['fullsid'] = df_act['sid'] + '_' + df_act['sample']
  df_act['fullsid'] = df_act.apply(sid_support, axis=1)



#  df_vol['sample'].replace('ADNI2/GO','ADNI2', inplace=True)   
#  df_vol['fullsid'] = df_vol['sid'] + '_' + df_vol['sample']
  df_vol['fullsid'] = df_vol.apply(sid_support2, axis=1)
  df_vol_temp =  df_vol[['fullsid', 'grp', 'sex1f', 'education_y' ,'age','MRI_field_strength','eTIV', 'MMSE']] 
  #df_temp = pd.merge(df_act, df_vol_temp, how='left', on='fullsid')
  df_temp = pd.merge(df_act, df_vol_temp, on='fullsid')
  
  #DROP scans for which we can not ascertain meta_features values based on the FastSurfer file we have atm
  #df_temp.dropna(subset = ['grp', 'sex1f','age','MRI_field_strength','eTIV', 'MMSE'],    
  df_temp.dropna(subset = ['grp', 'sex1f','age','MRI_field_strength','eTIV'],    
                 #Since AIBL does not have education_y info, we DON'T look for NAs in this column 
                 inplace=True)
  
  return df_temp




def scale_relevance_map(r_map_df, clipping_threshold=1, quantile=0.9999):
    """
    Path: Z:\Studien\Deep_Learning_Visualization\git-code\demenzerkennung\DeepLearningInteractiveVis\InteractiveVis\datamodel.py 
    Clips the relevance map to given threshold and adjusts it to range -1...1 float.

    :param numpy.ndarray relevance_map:
    :param int clipping_threshold: max value to be plotted, larger values will be set to this value
    :return: The relevance map, clipped to given threshold and adjusted to range -1...1 float.
    :rtype: numpy.ndarray
    """

    for index in r_map_df.index:

      r_map = r_map_df.iloc[index].values[3:]  

      # perform intensity normalization
      adaptive_relevance_scaling=True

      if adaptive_relevance_scaling:
          scale = np.quantile(np.absolute(r_map), quantile)  #.9999
      else:
          scale = 1/500     # multiply by 500
      if scale != 0:  # fallback if quantile returns zero: directly use abs max instead
          r_map = (r_map / scale)  # rescale range
      # corresponding to vmax in plt.imshow; vmin=-vmax used here
      # value derived empirically here from the histogram of relevance maps
      r_map[r_map > clipping_threshold] = clipping_threshold  # clipping of positive values
      r_map[r_map < -clipping_threshold] = -clipping_threshold  # clipping of negative values
      r_map = r_map / clipping_threshold  # final range: -1 to 1 float


      for i,col in enumerate(r_map_df.columns[3:]):
        r_map_df.iloc[0, col ] = r_map[i]  
    
    return r_map
