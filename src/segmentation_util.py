'''
A utility script, with different supporting functionalities.
23.05.24.
Author: Devesh Singh
'''

from keras import layers
from keras.layers import Input, Conv3D, BatchNormalization, Dense
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.layers import ReLU, concatenate
import keras.backend as K

import nibabel as nib
import numpy as np

import os
from os import listdir
from os.path import isfile,join


#Old setting
#x_range_from = 12
#x_range_to = 181
#y_range_from = 13
#y_range_to = 221
#z_range_from = 0
#z_range_to = 179
#minmaxscaling = False
#ip_shape = (179,169,208,1)

#-----------
##Our old Standard. LRP based.
Relevance_Method =[("lrp.sequential_preset_a", {"disable_model_checks":True, "neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0"), 
         # LRP CMP rule taken from https://github.com/berleon/when-explanations-lie/blob/master/when_explanations_lie.py
        ]    

#DeepLIFT amd IG. Found Useful by Bjarne.
#Relevance_Method =[  #("deeplift.DeepLIFTWrapper", {"neuron_selection_mode": "index"}, "DeepLIFT"), 
#    ('integrated_gradients', {}, 'IG')  
#        ]    

#DeepTaylor
#Relevance_Method =[  ('deep_taylor.bounded', {"neuron_selection_mode": "index", 'low':0, 'high':1}, 'DeepTaylor_Bounded' )
#        ]    


def get_sample_activation_IG(ig,sample_nifti,mean_cn_baseline,model,node):
    test_img = sample_nifti
    test_img = test_img.transpose((0,2,1))
    test_img = np.reshape(test_img, (1,)+ test_img.shape +(1,)) # add first subj index again to mimic original array structure
    
    baseline = mean_cn_baseline
    baseline = baseline.transpose((0,2,1))
    baseline = np.reshape(baseline, (1,)+ baseline.shape +(1,)) # add first subj index again to mimic original array structure
    
    #pred = model(test_img).numpy().argmax(axis=1)
    pred = node

    a = ig.explain(test_img, 
                    baselines = baseline,
                    target =  pred )     
    #if use models' actual predictions, we could dynamically chose the the node to create IG map from, based on the label/node it is most confident of.
    #OR as here, when target=pred=node, we could manually select one node to predict from.
    
    a = np.squeeze(a.attributions[0])
    a = a.transpose((0,2,1))  #Undoing previous transpose, to align the activations to segmentation maps--(seg_maps, are also o/p od read_nifti fun. so aligning with that).
               #we do not scale/clip/smooth the activation values. We insteas use he raw activcation values. 
    return a              
                    


def get_sample_activation(analyzer,sample_nifti,neuron):
  
    test_img = sample_nifti
    test_img = test_img.transpose((0,2,1))
    test_img = np.reshape(test_img, (1,)+ test_img.shape +(1,)) # add first subj index again to mimic original array structure
    
    a = analyzer.analyze(test_img, neuron_selection=neuron)
    a = np.squeeze(a)
    a = a.transpose((0,2,1))  #Undoing previous transpose, to align the activations to segmentation maps
               #we do not scale/clip/smooth the activation values. We insteas use he raw activcation values. 
    return a              
                        

#---------------
#paths to folders which contain the segementation files
seg_dirs =  {   'ADNI3': 'ADNI_t1linear/ADNI3',  
                'ADNI2' :{  'AD': 'ADNI_t1linear/AD',  
                            'CN': 'ADNI_t1linear/CN', 
                            'EMCI':'ADNI_t1linear/EMCI',
                            'LMCI':'ADNI_t1linear/LMCI'} , 
                'AIBL': 'AIBL_t1linear',
                'DELCODE': 'DELCODE_t1linear',
                'DESCRIBE': 'DESCRIBE_t1linear',
                'EDSD': 'EDSD_t1linear',
                'NIFD': {  'bvFTD': 'NIFD_t1linear/bvFTD',  
                            'CN':   'NIFD_t1linear/CN', 
                            'PFNA': 'NIFD_t1linear/PFNA',   
                            'SVD':  'NIFD_t1linear/SVD'},
                'OASIS': 'OASIS_t1linear',
}




filetype =   '{}_aparc.DKTatlas+aseg.deep.withCC_MNInew.nii.gz'        #has the segmasks from 'new'-er run of FastSurfer, ran in ~Sept'24.                

seg_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/data/' #'/mnt/data_dzne_archiv3/Studien/Deep_Learning_Visualization/data/'   

def find_sampleID(datasample_name):
    code_dict = datasample_name.split('_')
    cohort_code = code_dict[0]
    if 'OAS' in cohort_code:
        sid = code_dict[0].replace('OAS',"") + '_' + code_dict[1]
        cohort_code = 'OASIS'
    elif cohort_code in ['ADNI2', 'EDSD' ,'NIFD']:
        Diagnosis_OR_CenterCode =  code_dict[1]
        sid = Diagnosis_OR_CenterCode + '_' + code_dict[2]
    else:
        sid =  code_dict[1]
    return (cohort_code, sid)

def find_segemetnation_file(datasample_name):

    code_dict = datasample_name.split('_')
    cohort_code = code_dict[0]

    #FIND THE EXACT FILECODE (sid) OF THE SEGMENTATION FILE
    if ('OAS' in cohort_code) or (cohort_code == 'ADNI3'):
        #an outlier in naming convention, where the cohort_id is included while saving the segmentation file
        sid =  cohort_code + '_' + code_dict[1]                    
        if 'OAS' in cohort_code:
            cohort_code = 'OASIS'
    elif cohort_code in ['ADNI2', 'EDSD' ,'NIFD']:
        #Nested directory levels while saving segmentation files (which need the diagnosis/center code)
        Diagnosis_OR_CenterCode =  code_dict[1]
        if cohort_code == 'NIFD':
            sid = code_dict[2]
        else:    
            sid = Diagnosis_OR_CenterCode + '_' + code_dict[2]
    else:
        #AIBL; DELCODE; DESCRIBE
        sid =  code_dict[1]



    #FIND THE FULL PATH
    if cohort_code in ['ADNI2' ,'NIFD']:
        seg_path_ = join(seg_path,seg_dirs[cohort_code][Diagnosis_OR_CenterCode]) 
    else:
        seg_path_ = join(seg_path,seg_dirs[cohort_code])    #join(seg_path,seg_dirs2[cohort_code], '{}/mri/'.format(sid))  
    

    filename = filetype.format(sid)
    filepath = join(seg_path_,filename)

    return filepath


#---------------------------------------------------

# This label code dict taken from: https://github.com/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb
# The keys as are the ROIs and the values are the code that FastSurfer's segmentations produce (Think of them like a mask). 
# ex. pick 17 or 53 to mask everything other than L or R Hippocampus, respectively.
labels_lookup = {
'Left-Lateral-Ventricle': 4,
'Left-Inf-Lat-Vent': 5,
'Left-Cerebellum-White-Matter': 7,              
'Left-Cerebellum-Cortex': 8,
'Left-Thalamus-Proper': 10,
'Left-Caudate': 11,
'Left-Putamen': 12,
'Left-Pallidum': 13,
'Left-3rd-Ventricle': 14,
'Left-4th-Ventricle': 15,
'Left-Brain-Stem': 16,
'Left-Hippocampus': 17,                                      
'Left-Amygdala': 18,
'Left-CSF': 24,
'Left-Accumbens-area': 26,
'Left-VentralDC': 28,
'Left-choroid-plexus': 31,
'Right-Lateral-Ventricle': 43,
'Right-Inf-Lat-Vent': 44,
'Right-Cerebellum-White-Matter': 46,                      
'Right-Cerebellum-Cortex': 47,
'Right-Thalamus-Proper': 49,
'Right-Caudate': 50,
'Right-Putamen': 51,
'Right-Pallidum': 52,
'Right-Hippocampus': 53,                                  
'Right-Amygdala': 54,
'Right-Accumbens-area': 58,
'Right-VentralDC': 60,
'Right-choroid-plexus': 63,
'Right-3rd-Ventricle': 14,
'Right-4th-Ventricle': 15,
'Right-Brain-Stem': 16,
'Right-CSF': 24,
'ctx-lh-caudalanteriorcingulate': 1002,
'ctx-lh-caudalmiddlefrontal': 1003,
'ctx-lh-cuneus': 1005,
'ctx-lh-entorhinal': 1006,
'ctx-lh-fusiform': 1007,
'ctx-lh-inferiorparietal': 1008,
'ctx-lh-inferiortemporal': 1009,
'ctx-lh-isthmuscingulate': 1010,
'ctx-lh-lateraloccipital': 1011,
'ctx-lh-lateralorbitofrontal': 1012,
'ctx-lh-lingual': 1013,
'ctx-lh-medialorbitofrontal': 1014,
'ctx-lh-middletemporal': 1015,
'ctx-lh-parahippocampal': 1016,
'ctx-lh-paracentral': 1017,
'ctx-lh-parsopercularis': 1018,
'ctx-lh-parsorbitalis': 1019,
'ctx-lh-parstriangularis': 1020,
'ctx-lh-pericalcarine': 1021,
'ctx-lh-postcentral': 1022,
'ctx-lh-posteriorcingulate': 1023,
'ctx-lh-precentral': 1024,
'ctx-lh-precuneus': 1025,
'ctx-lh-rostralanteriorcingulate': 1026,
'ctx-lh-rostralmiddlefrontal': 1027,
'ctx-lh-superiorfrontal': 1028,
'ctx-lh-superiorparietal': 1029,
'ctx-lh-superiortemporal': 1030,
'ctx-lh-supramarginal': 1031,
'ctx-lh-transversetemporal': 1034,
'ctx-lh-insula': 1035,
'ctx-rh-caudalanteriorcingulate': 2002,
'ctx-rh-caudalmiddlefrontal': 2003,
'ctx-rh-cuneus': 2005,
'ctx-rh-entorhinal': 2006,
'ctx-rh-fusiform': 2007,
'ctx-rh-inferiorparietal': 2008,
'ctx-rh-inferiortemporal': 2009,
'ctx-rh-isthmuscingulate': 2010,
'ctx-rh-lateraloccipital': 2011,
'ctx-rh-lateralorbitofrontal': 2012,
'ctx-rh-lingual': 2013,
'ctx-rh-medialorbitofrontal': 2014,
'ctx-rh-middletemporal': 2015,
'ctx-rh-parahippocampal': 2016,
'ctx-rh-paracentral': 2017,
'ctx-rh-parsopercularis': 2018,
'ctx-rh-parsorbitalis': 2019,
'ctx-rh-parstriangularis': 2020,
'ctx-rh-pericalcarine': 2021,
'ctx-rh-postcentral': 2022,
'ctx-rh-posteriorcingulate': 2023,
'ctx-rh-precentral': 2024,
'ctx-rh-precuneus': 2025,
'ctx-rh-rostralanteriorcingulate': 2026,
'ctx-rh-rostralmiddlefrontal': 2027,
'ctx-rh-superiorfrontal': 2028,
'ctx-rh-superiorparietal': 2029,
'ctx-rh-superiortemporal': 2030,
'ctx-rh-supramarginal': 2031,
'ctx-rh-transversetemporal': 2034,
'ctx-rh-insula': 2035,
                     #Extra regions codes, which were found but are not present in this dict. Hence additions.
                     #link: https://www.sciencedirect.com/science/article/pii/S1053811922000623 (Tab.5, Column: FreeS)
                                          
'Left-Cerebral-White-Matter' : 2,
'Right-Cerebral-White-Matter': 41,
#New FS RUN. excludes[aseg.stats]  --   # Excluding Cortical Gray and White Matter    
                                        # ExcludeSegId 0 2 3 41 42 

'WM-hypointensities': 77,
'CC_Posterior': 251,           #CC: Corpus Callosum
'CC_Mid_Posterior': 252,      
'CC_Central': 253,    
'CC_Mid_Anterior': 254,   
'CC_Anterior': 255,                                
#'Exterioir':0,   #The outside of the brain area
}


#----------------------------------------------
def read_nifti_data(filename, x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling):
    """
    Reads the nifti file and loads the image data within the given field of view.

    Parameters
    ----------
    filename : string
        The input nifti file name.
    x_range_from : int >= 0
        Input file saggital slice start.
    x_range_to : int >= 0
        Input file saggital slice end.
    y_range_from : int >= 0
        Input file coronal slice start.
    y_range_to : int >= 0,
        Input file coronal slice end.
    z_range_from : int >= 0
        Input file axial slice start.
    z_range_to : int >= 0
        Input file axial slice end.
    minmaxscaling : bool
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1.

    Returns
    -------
    img : numpy array
        The loaded image data.

    """
    img = nib.load(filename)
    img = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
    img = np.transpose(img, (2, 0, 1)) # reorder dimensions to match coronal view z*x*y in MRIcron etc.
    img = np.flip(img) # flip all positions
    img = np.nan_to_num(img) # remove nan or inf values
    if minmaxscaling:
        img = (img - img.min()) / (img.max() - img.min()) # min/max scaling
        
    return img


def dir_check(save_path, non_empty_dir_ok=False):
    '''
    Create a director. And if it already exists, check if it is empty or not.
    save_path: a directory path
    '''
     
    os.makedirs(save_path, exist_ok=True)
    dir = os.listdir(save_path)
    if len(dir) == 0:
        #Empty directory
        pass
    else:
        #Not empty directory
        if non_empty_dir_ok:
            pass
        else:
            raise SystemExit('Not empty directory')




def sign_sum(arr, sign='pos'):
    sum = 0
    if sign=='pos':
        arr = np.where(arr<0, 0, arr)  #replace all negative value with 0
        sum = np.sum(arr)
        return sum
    else: #add up only the neg variables
        arr = np.where(arr>0, 0, arr)  #replace all positive value with 0
        sum = np.sum(arr)
        return sum



#------------- Model intialisation function calls -----------------------

def DenseNet(ip_shape, op_shape, filters = 3):
    '''
    declaring a DenseNet model. 
    Paper: https://arxiv.org/pdf/1608.06993.pdf
    Code adapted from https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8
    
    ip_shape: expected input shape
    op_shape: expected output shape
    filters: number of filters to be used
    '''
    
    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition=4):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, filters=8)
            y = bn_rl_conv(y, filters=8, kernel=3)
            x = concatenate([y,x])
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AveragePooling3D(2, strides = 2, padding = 'same')(x)
        return x
    


    input = Input(ip_shape)
    x = Conv3D(10, 7, strides = 2, padding = 'same')(input)
    x = MaxPooling3D(3, strides = 2, padding = 'same')(x)
    
    brc_in_blocks = [3,3]
    for repetition in brc_in_blocks:                      #[6,12,24,16]: 
        d = dense_block(x, repetition)
        x = transition_layer(d)
    
    #x = GlobalAveragePooling3D()(d)

    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
    #Notice the input to pooling layer. d. this overwrites the last x variable,
    #and nullifies the transition layer computations done on last dense blocks output. 
    # i.e last transition layer is not connected to the graph 
    #TLDR: No transition layer after last dense block. 

    # FC layer
    x=layers.Activation('relu')(x)
    #x=layers.Dropout(rate = 0.3)(x)
    x=layers.Dropout(rate = 0.4)(x)

    x = layers.Flatten()(x)
    output = Dense(op_shape, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model
