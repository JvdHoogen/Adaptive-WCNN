[PAGE IN PROGRESS]

### Adaptive-WCNN

This repository is supplementary to our paper "Classifying Multivariate Signals in Rolling Bearing Fault Detection Using Adaptive Wide-Kernel CNNs" published in MDPI Applied Sciences for reproducing our proposed models and their respective performance. In this repository, I will describe the models that I created plus the additional information on data processing and customized implementation on the proposed CNN architectures. For any questions, please feel free to contact me.





### Abstract
With the developments in improved computation power and the vast amount of (automatic) data collection, industry has become more data-driven. These data-driven 
approaches for monitoring processes and machinery require different modeling methods focusing on automated learning and deployment. In this context, deep learning 
provides possibilities for industrial diagnostics to achieve improved performance and efficiency. These deep learning applications can be used to automatically 
extract features during training, eliminating time-consuming feature engineering and prior understanding of sophisticated (signal) processing techniques. This paper 
extends on previous work, introducing one-dimensional (1D) CNN architectures that utilize an adaptive wide-kernel layer to improve classification of multivariate 
signals, e.g., time series classification in fault detection and condition monitoring context. We used multiple prominent benchmark datasets for rolling bearing 
fault detection to determine the performance of the proposed wide-kernel CNN architectures in different settings. For example, distinctive experimental conditions 
were tested with deviating amounts of training data. We shed light on the performance of these models compared to traditional machine learning applications and 
explain different approaches to handle multivariate signals with deep learning. Our proposed models show promising results for classifying different fault 
conditions of rolling bearing elements and their respective machine condition, while using a fairly straightforward 1D CNN architecture with minimal data 
preprocessing. Thus, using a 1D CNN with an adaptive wide-kernel layer seems well-suited for fault detection and condition monitoring. In addition, this paper 
clearly indicates the high potential performance of deep learning compared to traditional machine learning, particularly in complex multivariate and multi-class 
classification tasks.


### How to use
This repository contains a folder named "models". In this folder you will find two files. The files contain the full code for the Ada-WMCNN model and the other models used for our experiments. Preprocessing the data into sequences and a train and test set is necessary before utilizing the models. 

### Citation
When using our code, please cite our paper as follows:
```
@Article{app112311429,
AUTHOR = {van den Hoogen, Jurgen and Bloemheuvel, Stefan and Atzmueller, Martin},
TITLE = {Classifying Multivariate Signals in Rolling Bearing Fault Detection Using Adaptive Wide-Kernel CNNs},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {23},
ARTICLE-NUMBER = {11429},
URL = {https://www.mdpi.com/2076-3417/11/23/11429},
ISSN = {2076-3417},
DOI = {10.3390/app112311429}
}
```



### Requirements
Usage of our code requires many packages to be installed on your machine. The most important packages are listed below:
* Numpy
* Tensorflow
* Keras
* Multivariate-cwru


### Data
The data is collected from the [Paderborn University][paderborn] and the [Case Western Reserve University][cwru] that represents several bearing fault detection experiments based on multivariate signals.
For more information on the processing of the two datasets, please contact the author or consult the [`Multivariate CWRU`][multivariate_cwru] package description to extract and preprocess for the CWRU dataset. 




[cwru]: <https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website>
[multivariate_cwru]: <https://github.com/JvdHoogen/multivariate_cwru>
[paderborn]: <https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download>


