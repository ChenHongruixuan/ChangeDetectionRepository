# Change Detection Method


## Traditional Methods
### Change Vector Analysis (CVA)
Change vector analysis (CVA) [1] is a most commonly used method, which can provide change intensity and change direction. 

### Slow Feature Analysis (SFA)
Wu et al. [2] proposed a novel CD method based on slow feature analysis (SFA), which aims to find the most invariant component in multitemporal images to highlight changed regions. In addition to change detection, SFA was also used in radiometric correction [3] and scene change detection [4].

### Multivariate Alteration Detection (MAD)
MAD is a change detection algorithm based on canonical correlation analysis (CCA) that aims to maximize the variance of projection feature difference. For the detailed introduction about MAD, please refer to [5] and [6]. 

## Deep Learning Methods
### Deep Slow Feature Analysis (DSFA)
DSFA is an unsupervised change detection model that utilizes a dual-stream deep neural network to learn non-linear features and highlights changes via linear SFA. For the detailed introduction about DSFA, please refer to [7]. And the tensorflow implementation of DSFA can be founded in https://github.com/rulixiang/DSFANet. 

### Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (SiamCRNN)

### Deep Kernel PCA Convolutional Mapping Network (KPCA-MNet)
