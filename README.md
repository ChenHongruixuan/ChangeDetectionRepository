# Change Detection Repository
In this repository, we provide python implementation of some traditional change detection methods, such as SFA, MAD, some deep learning-based change detection methods, such as SiamCRNN, DSFA, and FCN-based methods, or their original websites. Some multi-temporal data sets are also contained in this repository. We would be very glad if this repository can provide some help to your research in change detection or remote sensing image interpretation.


## Traditional Methods
### Change Vector Analysis (CVA)
Change vector analysis (CVA) [1] is a most commonly used method, which can provide change intensity and change direction. 

### Slow Feature Analysis (SFA)
<div align=center><img src="./Figure/SFA.png" width="60%" height="60%"></div>
Wu et al. [2] proposed a novel CD method based on slow feature analysis (SFA), which aims to find the most invariant component in multitemporal images to highlight changed regions. In addition to change detection, SFA was also used in radiometric correction [3] and scene change detection [4]. This reporisty contains the Python implementation of SFA and iterative SFA. The MATLAB implementation can be founded in http://sigma.whu.edu.cn/resource.php. 

### Multivariate Alteration Detection (MAD)
MAD is a change detection algorithm based on canonical correlation analysis (CCA) that aims to maximize the variance of projection feature difference. For the detailed introduction about MAD, please refer to [5] and [6]. This reporisty contains the python implementation of MAD. The MATLAB implementation can be founded in http://www.imm.dtu.dk/~alan/software.html. 

### PCA-Kmeans
<div align=center><img src="./Figure/PCA_Kmeans.png" width="50%" height="50%"></div>
PCA-Kmeans [12] partitones the difference image into nonoverlapping blocks. Orthonormal eigenvectors are extracted through PCA of nonoverlapping block set to create an eigenvector space. Each pixel in the difference image is represented with an S-dimensional feature vector which is the projection difference image data onto the generated eigenvector space. The change detection is achieved by partitioning the feature vector space into two clusters using k-means. 

## Deep Learning Methods
### Deep Slow Feature Analysis (DSFA)
<div align=center><img src="./Figure/DSFA.png" width="60%" height="60%"></div>
DSFA is an unsupervised change detection model that utilizes a dual-stream deep neural network to learn non-linear features and highlights changes via linear SFA. For the detailed introduction about DSFA, please refer to [7]. The Tensorflow implementation of DSFA can be founded in https://github.com/rulixiang/DSFANet or http://sigma.whu.edu.cn/resource.php. 

### Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (SiamCRNN)
<div align=center><img src="./Figure/SiamCRNN.png" width="70%" height="70%"></div>
SiamCRNN is an end-to-end general multi-source change detection architecture that consists of three subnetworks: deep siamese convolutional neural network (DSCNN), multiple-layers RNN (MRNN), and fully connected (FC) layers. The DSCNN has a flexible structure for multisource image and is able to extract spatial–spectral features from homogeneous or heterogeneous VHR image patches. The MRNN stacked by long-short term memory (LSTM) units is responsible for mapping the spatial–spectral features extracted by DSCNN into a new latent feature space and mining the change information between them. In addition, FC, the last part of SiamCRNN, is adopted to predict change probability. For the detailed introduction about DSFA, please refer to [8]. The Tensorflow implementation of SiamCRNN can be founded in https://github.com/I-Hope-Peace/SiamCRNN.

### Deep Kernel PCA Convolutional Mapping Network (KPCA-MNet)
<div align=center><img src="./Figure/KPCAMNet.png" width="70%" height="70%"></div>
KPCA-MNet is designed for unsupervised binary and multi-class change detection in very-high-resolution images. In the KPCA-MNet, the high-level spatial-spectral feature maps are extracted by a deep siamese network consisting of weight-shared KPCA convolutional layers. Then, the change information in the feature difference map is mapped into a 2-D polar domain. Finally, the change detection results are generated by threshold segmentation and clustering algorithms. For the detailed introduction about DSFA, please refer to [9]. The Python implementation will be released soon. 

### SARPCANet
<div align=center><img src="./Figure/SAR_PCANet_1.png" width="60%" height="60%"></div>
SARPCANet utilizes Gabor wavelets and FCM as the pre-classification method to select training samples [10], and then trains a PCANet [11] model with the selected image patches. The original MATLAB implementation could be founded in https://github.com/summitgao/SAR_Change_Detection_GarborPCANet. 

### FDCNN
<div align=center><img src="./Figure/FDCNN.png" width="60%" height="60%"></div>
FDCNN [13] uses scene-level samples of remote sensing scene classification for learning deep features from different remote sensing scenes at different scales. Then, a new CNN structure and training strategies are proposed for remote sensing image change detection, which is supervised but requires very few pixel-level training samples.The original Caffe implementation could be founded in https://github.com/MinZHANG-WHU/FDCNN. 



## Reference
[1] F. Bovolo and L. Bruzzone, “A Theoretical Framework for Unsupervised Change Detection Based on Change Vector Analysis in the Polar Domain,” IEEE Trans. Geosci. Remote Sens., vol. 45, no. 1, pp. 218–236, 2007.  
[2] C. Wu, B. Du, and L. Zhang, “Slow feature analysis for change detection in multispectral imagery,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 5, pp. 2858–2874, 2014.  
[3] L. Zhang, C. Wu, and B. Du, “Automatic radiometric normalization for multitemporal remote sensing imagery with iterative slow feature analysis,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 10, pp. 6141–6155, 2014.  
[4] C. Wu, L. Zhang, and B. Du, “Kernel Slow Feature Analysis for Scene Change Detection,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 4, pp. 2367–2384, 2017.  
[5] A. A. Nielsen, K. Conradsen, and J. J. Simpson, “Multivariate alteration detection (MAD) and MAF Postprocessing in multispectral, bitemporal image data: New approaches to change detection studies,” Remote Sens. Environ., vol. 64, pp. 1–19, 1998.  
[6] A. A. Nielsen, “The regularized iteratively reweighted MAD method for change detection in multi- and hyperspectral data,” IEEE Trans. Image Process., vol. 16, no. 2, pp. 463–478, 2007.  
[7] B. Du, L. Ru, C. Wu, and L. Zhang, “Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 12, pp. 9976–9992, 2019.  
[8] H. Chen, C. Wu, B. Du, L. Zhang, and L. Wang, “Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network,” IEEE    Trans. Geosci. Remote Sens., vol. 58, no. 4, pp. 2848–2864, 2020.  
[9] C. Wu,  H. Chen, B. Do, and L. Zhang, “Unsupervised Change Detection in Multi-temporal VHR Images Based on Deep Kernel PCA Convolutional Mapping Network,” arXiv preprint arXiv:1912.08628, 2019. https://arxiv.org/abs/1912.08628v1.  
[10] F. Gao, J. Dong, B. Li, and Q. Xu, “Automatic Change Detection in Synthetic Aperture Radar Images Based on PCANet,” IEEE Geosci. Remote Sens. Lett., vol. 13, no. 12, pp. 1792–1796, 2016.  
[11] T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, “PCANet: A Simple Deep Learning Baseline for Image Classification?,” IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017–5032, 2015.  
[12] T. Celik, “Unsupervised change detection in satellite images using principal component analysis and K-means clustering,” IEEE Geosci. Remote Sens. Lett., vol. 6, no. 4, pp. 772–776, 2009.  
[13] M. Zhang and W. Shi, “A Feature Difference Convolutional Neural Network-Based Change Detection Method,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 10, pp. 7232–7246, 2020.
## Q & A
**For any questions, please [contact us.](mailto:Qschrx@gmail.com)**
