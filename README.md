# Change Detection Method


## Traditional Methods
### Change Vector Analysis (CVA)
Change vector analysis (CVA) [1] is a most commonly used method, which can provide change intensity and change direction. 

### Slow Feature Analysis (SFA)
Wu et al. [2] proposed a novel CD method based on slow feature analysis (SFA), which aims to find the most invariant component in multitemporal images to highlight changed regions. In addition to change detection, SFA was also used in radiometric correction [3] and scene change detection [4]. This reporisty contains the python implementation of SFA and iterative SFA. The MATLAB implementation can be founded in http://sigma.whu.edu.cn/resource.php. 

### Multivariate Alteration Detection (MAD)
MAD is a change detection algorithm based on canonical correlation analysis (CCA) that aims to maximize the variance of projection feature difference. For the detailed introduction about MAD, please refer to [5] and [6]. This reporisty contains the python implementation of MAD. The MATLAB implementation can be founded in http://www.imm.dtu.dk/~alan/software.html. 

## Deep Learning Methods
### Deep Slow Feature Analysis (DSFA)
DSFA is an unsupervised change detection model that utilizes a dual-stream deep neural network to learn non-linear features and highlights changes via linear SFA. For the detailed introduction about DSFA, please refer to [7]. The tensorflow implementation of DSFA can be founded in https://github.com/rulixiang/DSFANet or http://sigma.whu.edu.cn/resource.php. 

### Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (SiamCRNN)
SiamCRNN is an end-to-end general multi-source change detection architecture that consists of three subnetworks: deep siamese convolutional neural network (DSCNN), multiple-layers RNN (MRNN), and fully connected (FC) layers. The DSCNN has a flexible structure for multisource image and is able to extract spatial–spectral features from homogeneous or heterogeneous VHR image patches. The MRNN stacked by long-short term memory (LSTM) units is responsible for mapping the spatial–spectral features extracted by DSCNN into a new latent feature space and mining the change information between them. In addition, FC, the last part of SiamCRNN, is adopted to predict change probability. For the detailed introduction about DSFA, please refer to [8]. The tensorflow implementation of SiamCRNN can be founded in https://github.com/I-Hope-Peace/SiamCRNN.

### Deep Kernel PCA Convolutional Mapping Network (KPCA-MNet)

## Reference
[1] F. Bovolo and L. Bruzzone, “A Theoretical Framework for Unsupervised Change Detection Based on Change Vector Analysis in the Polar Domain,” IEEE Trans. Geosci. Remote Sens., vol. 45, no. 1, pp. 218–236, 2007.
[2] C. Wu, B. Du, and L. Zhang, “Slow feature analysis for change detection in multispectral imagery,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 5, pp. 2858–2874, 2014.
[3] L. Zhang, C. Wu, and B. Du, “Automatic radiometric normalization for multitemporal remote sensing imagery with iterative slow feature analysis,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 10, pp. 6141–6155, 2014.
[4] C. Wu, L. Zhang, and B. Du, “Kernel Slow Feature Analysis for Scene Change Detection,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 4, pp. 2367–2384, 2017.
[5] A. A. Nielsen, K. Conradsen, and J. J. Simpson, “Multivariate alteration detection (MAD) and MAF Postprocessing in multispectral, bitemporal image data: New approaches to change detection studies,” Remote Sens. Environ., vol. 64, pp. 1–19, 1998.
[6] A. A. Nielsen, “The regularized iteratively reweighted MAD method for change detection in multi- and hyperspectral data,” IEEE Trans. Image Process., vol. 16, no. 2, pp. 463–478, 2007.
[7] B. Du, L. Ru, C. Wu, and L. Zhang, “Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 12, pp. 9976–9992, 2019.
[8] H. Chen, C. Wu, B. Du, L. Zhang, and L. Wang, “Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network,” IEEE    Trans. Geosci. Remote Sens., vol. 58, no. 4, pp. 2848–2864, 2020.

## Q & A
**For any questions, please [contact us.](mailto:Qschrx@gmail.com)**
