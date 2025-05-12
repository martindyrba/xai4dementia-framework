# xai4dementia-framework
An Unsupervised Explainable AI Framework for Dementia Detection with Context Enrichment

![Python](https://img.shields.io/badge/Python-v3.11-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15.1-orange)
![iNNvestigate](https://img.shields.io/badge/iNNvestigate-v2.1.2-blue)


Explainable Artificial Intelligence (XAI) methods enhance the diagnostic efficiency of clinical decision support systems by making the predictions of a convolutional neural network’s (CNN) on brain imaging more transparent and trustworthy. However, their clinical adoption is hindered due to the limited validation of the explanation quality. Our study introduces a framework that evaluates XAI methods by integrating neuroanatomical morphological features - gray matter volumetry and average cortical thickness signals, with CNN-generated relevance maps for disease classification.


## Pipeline Overview
The workflow of our study is schematically presented in Figure below. Our framework provides several ways to generate post-hoc explanations for a CNN model trained to detect dementia diseases, including: i) global-level explanations, such as membership in the stable versus converter subgroups, and ii) local-level explanations for each individual prediction, such as ii-a) example-based explanations of cognitive trajectories or ii-b) textual explanation by pathology summarization.

<p align="center">
  <img src="/images/1.png" style="width:100%; max-width:100%;">
</p>
<![Pipeline Flow](/images/1.png)>

