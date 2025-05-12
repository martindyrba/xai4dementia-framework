import pandas as pd
import numpy as np

import matplotlib 
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc
import scipy
import os
from datetime import datetime

import tensorflow as tf
print(tf.version.VERSION)
tf.compat.v1.disable_eager_execution() ## required by innvestigate to be able to analyze the network graph
print("Running TensorFlow version: ", tf.version.VERSION)
from keras.losses import CategoricalFocalCrossentropy
from keras import layers
from keras.layers import Input, Conv3D, BatchNormalization, Dense
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.layers import ReLU, concatenate
import keras.backend as K
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from aucmedi.data_processing.io_data import input_interface
from aucmedi.evaluation import performance
from aucmedi.data_processing import data_generator
from aucmedi.data_processing.io_loader.sitk_loader import sitk_loader
from aucmedi.sampling import sampling_kfold
from aucmedi.sampling.split import sampling_split
from aucmedi.data_processing import augmentation
from aucmedi.utils.class_weights import compute_class_weights

import nibabel as nib
import innvestigate


##-----------------------META CONSIDERATIONS-------------------------------------------------------------

# Get current date and time
current_time = datetime.now()
# Convert to string in the format: 'YYYY-MM-DD HH:MM'
time_str = current_time.strftime('%Y-%m-%d_%H%M')
exp_name = time_str + '_' + 'k-fold_earlystrop_MCIcorrect'
 

# limit memory consumption to allocate only what is currently needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    
    #Chose only one GPU for model training.
    GPU_Number = 0   
    tf.config.set_visible_devices(gpus[GPU_Number], 'GPU')
    #tf.config.set_visible_devices(gpus[:], 'GPU')     #use all GPUs
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


path_imagedir='/ssd1/dyrba/xxx_t1linear_N4cor_WarpedSegmentationPosteriors2_allinone'                  # Path to images

# Load the image names and labels info from the csv file
nifti_loader = input_interface(
    interface="csv", # Type of annotations
    # image_format=".nii.gz",
    path_imagedir=path_imagedir,                  # Path to images
    path_data='./data/xxx_t1linear_master_EXPProject.csv',                                                  # Path to labels
    col_sample="Filename",                                                                                  # Name of the column that identifies an image
    col_class="derived_grp",                                                                                # Name of the column that classify the image
    training=True,                                                                                          # Is annotation data available?
    ohe=False,                                                                                              # Are labels one-hot-encoded?
)
(nifti_images, labelsbin, n_classes, class_names, image_format) = nifti_loader
df_master_data = pd.read_csv('./data/xxx_t1linear_master_EXPProject.csv')
meta_cols = ['Sample','Original Diagnosis grp','Sex1F','Age','Education','MRI Field Strength','ETIV','MMSE','cdr']
meta_grp = df_master_data[df_master_data['Filename'].isin(nifti_images)][['Filename'] + meta_cols]
meta_grp['Filename'] = pd.Categorical(meta_grp['Filename'], categories=nifti_images, ordered=True)
meta_grp = meta_grp.sort_values('Filename')


#Model description
def DenseNet(ip_shape, op_shape, filters = 5):
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

   #batch norm + relu + conv
   def rl_bn_conv(x,filters,kernel=1,strides=1):
       x = ReLU()(x)
       x = BatchNormalization()(x)
       x = Conv3D(filters, kernel, strides=strides,padding = 'same')(x)
       return x



   def dense_block(x, repetition=4):

       for _ in range(repetition):
           #y = rl_bn_conv(x, filters=8)
           #y = rl_bn_conv(y, filters=8, kernel=3)
           y = bn_rl_conv(x, filters=8)
           y = bn_rl_conv(y, filters=8, kernel=3)

           x = concatenate([y,x])
       return x

   def transition_layer(x):

       #x = rl_bn_conv(x, K.int_shape(x)[-1] //2 )
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

   x = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
   #Notice the input to the pooling layer. d. This overwrites the last x variable,
   #and nullifies the transition layer computations done on the last dense blocks output.
   # i.e, last transition layer is not connected to the graph
   #TLDR: No transition layer after last dense block.

   # FC layer
   x=layers.Activation('relu')(x)
   x=layers.Dropout(rate = 0.4)(x)

   x = layers.Flatten()(x)
   output = Dense(op_shape, activation = 'softmax', kernel_regularizer='l1_l2')(x)

   model = Model(input, output)
   return model



#define nifti datei 
def read_nifti_data(filename, x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling):
    """
    Reads the NIFTI file and loads the image data within the given field of view.

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


def dir_check(save_path):
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
        raise SystemExit('Not empty relevance directory')


def save_scores_to_dataframe(test_labels, test_images, GD_study_labels,class_names, predictions):
    # Convert one-hot encoded labels to class names
    true_classes = np.argmax(test_labels, axis=1)     #derived
    true_labels = [class_names[i] for i in true_classes]

    # Convert CNN predictions (softmax outputs) to predicted class names
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = [class_names[i] for i in predicted_classes]



    # Create a DataFrame with the true and predicted labels and the softmax probabilities
    df = pd.DataFrame({
        'AD Probability': predictions[:, 0],
        'CN Probability': predictions[:, 1],
        'FTD Probability': predictions[:, 2],
        'Predicted Label': predicted_labels,
        'DerivedLabels(Used4Training)': true_labels,     #USED DURING TRAINING
        'GD_StudyLabel(TakenFromCohortsExls)': GD_study_labels,
        'Filename': test_images
    })

    return df




###---------------------------------------TRAINING SETUP-----------------------------------------------

# Define the evaluation metrics to compute 
#When the 2nd class is not specified, we assume that the +ve class is mentioned and the -ve class is CN
acc_AD, acc_FTD, acc_ADFTD, acc_MCI = [], [], [], []         
auc_AD, auc_FTD, auc_ADFTD, auc_MCI = [], [], [], []

n_splits = 5 
learning_rate = 0.0001
num_epochs = 100 
splits = [0.9, 0.1]         
seed = 42
workers = 32         
batch = 128      


x, y, z = 193, 229, 193
numfiles = len(nifti_images)
input_shape = (numfiles, x, y, z, 1)


testimg_path = "/ssd1/dyrba/xxx_t1linear_N4cor_WarpedSegmentationPosteriors2_allinone/ADNI3_6849_N4cor_WarpedSegmentationPosteriors2.nii.gz"
test_img = read_nifti_data(testimg_path, 0, 193, 0, 229, 0, 193, minmaxscaling=True)
#print(grps.iloc[0, :]) # print subject information
test_img = test_img.transpose((0,2,1))
test_img = np.reshape(test_img, (1,)+ test_img.shape +(1,)) # add first subj index again to mimic original array structure


cv = sampling_kfold(
        samples=nifti_images, # list of sample names
        labels=labelsbin, # list of corresponding labels
        metadata=meta_grp, # metadata as numpy array)
        n_splits = n_splits,
        stratified=True, 
        iterative=False, # Approximation of a solution is only useful for multi-label problems
        seed= seed 

)

# activate flip and scale augmentation
data_aug = augmentation.VolumeAugmentation(flip=True, rotate=False, brightness=False, contrast=False,
                saturation=False, hue=False, scale=True, crop=False,
                grid_distortion=False, compression=False, gaussian_noise=False,
                gaussian_blur=False, downscaling=False, gamma=False,
                elastic_transform=False)
    

#The K-fold cross-validation loop starts here.
for k,fold in enumerate(cv):
    savepath= './train/{}/cv_{}'.format(exp_name, k+1)
    dir_check(savepath)

    early_stopping = EarlyStopping(
                        monitor='val_loss', 
                        mode='auto', 
                        verbose=0, 
                        patience=5, 
                        min_delta=0.01)

    model_checkpoint = ModelCheckpoint(
        filepath=(os.path.join(savepath,"ADMCIvsCNvsFTD_cv{:02d}.h5".format(k+1))),
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True
    )
    
    (train_x, train_y, train_m, test_x, test_y, test_m) = fold

    set_train, set_val = sampling_split(
        samples=train_x,           # list of sample names
        labels=train_y,               # list of corresponding labels
        metadata=train_m,              # metadata as numpy array        
        sampling=splits, # percentage splits
        stratified=True, 
        iterative=False, # Approximation of a solution is only useful for multi label problems
        seed= seed+k 
    )

    (train_x, train_y, train_m) = set_train
    (val_x, val_y, val_m) = set_val

    gen_train = data_generator.DataGenerator(
        samples=train_x,
        labels=train_y,
        metadata=train_m,
        path_imagedir=path_imagedir, # Path to images
        image_format=None, #subfunctions=[], #image format is read automatically
        loader=sitk_loader, # Loader function for .nii image format files
        batch_size=batch,          
        workers=workers,             
        grayscale=True,
        data_aug=data_aug, # set 'None' to disable augmentations
        resize=(193,229,193), #full size of nifti image (x,y,z)
        standardize_mode="minmax", #normalization method
        seed=seed+k
    )
    
    # validation generator without augmentation
    gen_val = data_generator.DataGenerator(
        samples=val_x,
        labels=val_y,
        metadata=val_m,
        path_imagedir=path_imagedir, # Path to images
        image_format=None, #subfunctions=[], #image format is read automatically
        loader=sitk_loader, # Loader function for .nii image format files
        batch_size=batch,
        workers=workers,
        grayscale=True,
        data_aug=None, # set 'None' to disable augmenations
        resize=(193,229,193), #full size of nifti image (x,y,z)
        standardize_mode="minmax", #normalization method
        seed= seed+k
    )
        
    # Create a Test generator, w/o augmentations                                              
    gen_test = data_generator.DataGenerator(
    samples=test_x,
    labels=test_y,
    metadata=test_m,
    path_imagedir=path_imagedir, # Path to images
    image_format=None, #subfunctions=[], #image format is read automatically
    loader=sitk_loader, # Loader function for .nii image format files
    batch_size=batch,
    workers=workers,
    grayscale=True,
    data_aug=None,
    resize=(193,229,193), #full size of nifti image (x,y,z)
    standardize_mode="minmax", #normalization method
    seed=seed+k
    )
    
    #Initialize model
    cw_loss, cw_fit = compute_class_weights(train_y)
    model = DenseNet(input_shape[1:], labelsbin.shape[1])
    opt = tf.keras.optimizers.legacy.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
#    model.compile(loss=CategoricalFocalCrossentropy(alpha = cw_loss), optimizer = opt, metrics=['accuracy'])
    model.summary()

    #fit model 
    history = model.fit(
        x=gen_train,
        epochs=num_epochs,
        verbose=1,
        validation_data=gen_val,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=cw_fit  
    )
    
    # --- plot model's statistics from training -----
    loss = history.history['loss'] #training
    val_loss = history.history['val_loss']  #validation
    
    acc = history.history['accuracy'] #training acc
    val_acc = history.history['val_accuracy'] #validation

    epochsr = range(1, len(loss)+1, 1)    #update, as the number of epochs trained could be less than the max of 50 due to early stopping.
    plt.figure()
    plt.plot(epochsr, loss, 'bo', label='Training loss')
    plt.plot(epochsr, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(savepath,'Loss_train.png'))       
    #plt.show()

    plt.figure()
    plt.plot(epochsr, acc, 'bo', label='Training acc')
    plt.plot(epochsr, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy')   
    plt.legend()
    plt.savefig(os.path.join(savepath,'Acc_train.png'))
    #plt.show()


    #------ PLOT relevances for a test sample ----
    mymodel = load_model(os.path.join(savepath,"ADMCIvsCNvsFTD_cv{:02d}.h5".format(k+1)))  
    methods = [ # tuple with method,     params,                  label
            ("lrp.sequential_preset_a", {"disable_model_checks":True, "neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0"), 
            # LRP CMP rule taken from https://github.com/berleon/when-explanations-lie/blob/master/when_explanations_lie.py
    ]

    # replace last layer's softmax activation function by linear
    model_wo_softmax = innvestigate.model_wo_softmax(mymodel)

    # create analyzer
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
        analyzers.append(analyzer)
        

    for method,analyzer in zip(methods, analyzers):
        for neuron in [0,1]:                                                  
            
            savepath_temp = os.path.join(savepath, 'relevance/neuron_{}'.format(neuron))
            dir_check(savepath_temp)

            anal = analyzer.analyze(test_img, neuron_selection=neuron)
            for ch in [0]:
                a = anal[0,:,:,:,ch] # first axis is subject
                a = np.squeeze(a)
                a = scipy.ndimage.filters.gaussian_filter(a, sigma=0.8) # smooth activity image

                amax = np.quantile(np.abs(a), 0.9999) # np.amax(a)
                amin = -amax # np.amin(a)
                a = a/amax # range: [-1,1]
                bg = np.squeeze(test_img[0,:,:,:,ch]) # first axis is subject

                alpha_mask = np.int8(np.abs(a)>0.1)
                for i in range(test_img.shape[2]):
                    if (i % 3 == 0): # only display each xx slice
                        plt.figure()
                        plt.imshow(bg[:,i,:], cmap='gray')
                        plt.imshow(a[:,i,:], cmap='jet', alpha=alpha_mask[:,i,:], vmin=-1, vmax=1) # cmap='hot'
                        plt.savefig(os.path.join(savepath_temp,'act_testsample_{}.png'.format(i)))
                        #plt.show()

    #---- Test evaluations -----
    pred = mymodel.predict(gen_test)                
    df = save_scores_to_dataframe(test_y, test_x, test_m[:,2], class_names, pred)
    df.to_csv(os.path.join(savepath,'test_set_scores_{}.csv'.format(k+1)), index=False)

    #Although, we clubbed AD and MCI for training to make it easier for model to find pathology, we want to test these classes seprately.
    #OR rather, we want to test the predictions for true pure AD samples without w/o MCI samples 
    MCI_indices = [i for i, x in enumerate(test_m[:,2]) if x == 'MCI']
    pred_wo_MCI = np.delete(pred, MCI_indices, axis=0)
    test_y_wo_MCI = np.delete(test_y, MCI_indices, axis=0)

    savepath_temp=os.path.join(savepath, 'AD-CN-FTD')
    dir_check(savepath_temp)
    performance.evaluate_performance(
        preds=pred_wo_MCI,
        labels=test_y_wo_MCI,                        #set_test[1]
        out_path=savepath_temp,
        class_names=class_names,
        show=False,                               #True
    )                    


    #Generate binarised metrics
    #test_y  --> test_y_wo_MCI;    pred --> pred_wo_MCI
    bool_AD_CN  = test_y_wo_MCI[:,0] | test_y_wo_MCI[:,1]                #a list of all the AD_CN samples    #0:AD, 1:CN, 2:FTD
    bool_FTD_CN = test_y_wo_MCI[:,2] | test_y_wo_MCI[:,1]                #a list of all the FTD_CN samples  
    bool_AD_FTD = test_y_wo_MCI[:,0] | test_y_wo_MCI[:,2]                #a list of all the AD_FTD samples  
    set_test_FTD_CN = test_y_wo_MCI[bool_FTD_CN]   
    pred_FTD_CN = pred_wo_MCI[bool_FTD_CN]
    set_test_AD_CN = test_y_wo_MCI[bool_AD_CN]
    pred_AD_CN = pred_wo_MCI[bool_AD_CN]
    set_test_AD_FTD = test_y_wo_MCI[bool_AD_FTD]   
    pred_AD_FTD = pred_wo_MCI[bool_AD_FTD]



   # ----Evaluation metrics for AD (positive) vs CN(negative)----
    savepath_temp = os.path.join(savepath, 'AD-CN')
    dir_check(savepath_temp)
    fpr_AD_CN, tpr_AD_CN, _ = roc_curve(set_test_AD_CN[:, 0], pred_AD_CN[:, 0])
    roc_auc_AD_CN = auc(fpr_AD_CN, tpr_AD_CN)
    acc_AD_CN = np.mean((set_test_AD_CN[:, 0] == np.round(pred_AD_CN[:, 0])).astype(int))*100
    
    print('AUC for AD vs. CN = %0.3f' % roc_auc_AD_CN)
    print('Acc for AD vs. CN = %0.1f' % acc_AD_CN)
    auc_AD.append(roc_auc_AD_CN)
    acc_AD.append(acc_AD_CN)

    performance.evaluate_performance(
        preds=pred_AD_CN[:, [0,1]],
        labels=set_test_AD_CN[:,[0,1]],
        out_path=savepath_temp,
        class_names=[class_names[i] for i in [0,1]],
        show=False,
        #multi_label=is_multilabel
    )                     

    #---- Evaluation metrics for FTD (positive) vs CN (negative)----
    savepath_temp = os.path.join(savepath, 'FTD-CN')
    dir_check(savepath_temp)   
    fpr_FTD_CN, tpr_FTD_CN, _ = roc_curve(set_test_FTD_CN[:, 2], pred_FTD_CN[:, 2])
    roc_auc_FTD_CN = auc(fpr_FTD_CN, tpr_FTD_CN)
    acc_FTD_CN = np.mean((set_test_FTD_CN[:, 2] == np.round(pred_FTD_CN[:, 2])).astype(int))*100

        
    print('AUC for FTD vs. CN = %0.3f' % roc_auc_FTD_CN)
    print('Acc for FTD vs. CN = %0.1f' % acc_FTD_CN)
    auc_FTD.append(roc_auc_FTD_CN)
    acc_FTD.append(acc_FTD_CN)
    performance.evaluate_performance(
        preds=pred_FTD_CN[:, [1,2]],
        labels=set_test_FTD_CN[:, [1,2]],
        out_path=savepath_temp,
        class_names=[class_names[i] for i in [1,2]],
        show=False,
    )

    # ---- Evaluation metrics for AD(positive) vs FTD(negative) ----   
    savepath_temp = os.path.join(savepath, 'AD-FTD')
    dir_check(savepath_temp)   
    fpr_AD_FTD, tpr_AD_FTD, _ = roc_curve(set_test_AD_FTD[:, 0], pred_AD_FTD[:, 0])
    roc_auc_AD_FTD = auc(fpr_AD_FTD, tpr_AD_FTD)
    acc_AD_FTD = np.mean((set_test_AD_FTD[:, 0] == np.round(pred_AD_FTD[:, 0])).astype(int))*100

        
    print('AUC for AD vs. FTD = %0.3f' % roc_auc_AD_FTD)
    print('Acc for AD vs. FTD = %0.1f' % acc_AD_FTD)
    auc_ADFTD.append(roc_auc_AD_FTD)
    acc_ADFTD.append(acc_AD_FTD)
    performance.evaluate_performance(
        preds=pred_AD_FTD[:,[0,2]] ,
        labels=set_test_AD_FTD[:,[0,2]],
        out_path=savepath_temp,
        class_names=[class_names[i] for i in [0,2]],
        show=False,
    )

    #---- Plotting binarized ROC curves ------
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # diagonal as reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr_FTD_CN, tpr_FTD_CN, color='darkorange', label='ROC curve FTD vs. CN (area = %0.2f)' % roc_auc_FTD_CN)
    plt.plot(fpr_AD_CN, tpr_AD_CN, color='red', label='ROC curve AD vs. CN (area = %0.2f)' % roc_auc_AD_CN)
    plt.plot(fpr_AD_FTD, tpr_AD_FTD, color='blue', label='ROC curve AD vs. FTD (area = %0.2f)' % roc_auc_AD_FTD)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(savepath,'ROC.png'))

#Save the metrics from all folds
df_metrics = pd.DataFrame()
df_metrics['fold'] = [i for i in range(1, n_splits+1)]  
df_metrics['Acc_ADvsCN'] = acc_AD 
df_metrics['Acc_FTDvsCN'] = acc_FTD 
df_metrics['Acc_ADvsFTD'] = acc_ADFTD 
df_metrics['AUC_ADvsCN'] = auc_AD 
df_metrics['AUC_FTDvsCN'] = auc_FTD 
df_metrics['AUC_ADvsFTD'] = auc_ADFTD 
df_metrics.to_csv(os.path.join('./train/{}/'.format(exp_name),'test_set_metrics.csv'), index=False)



    

print()
