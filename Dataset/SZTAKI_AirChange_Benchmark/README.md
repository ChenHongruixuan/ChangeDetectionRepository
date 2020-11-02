# SZTAKI AirChange Benchmark set
## Description
This Benchmark set contains 13 aerial image pairs of size 952x640 and resolution 1.5m/pixel and binary change masks (drawn by hand), which were used for evaluation in publications [1] and [2].
Each record constains a pair of preliminary registered input images and a mask of the 'relevant' changes. The input images are taken with 5, 7 resp. 23 years time differences. During the generation of the change mask, we have considered the following differences as relevant changes: (a) new built-up regions (b) building operations (c) planting of large group of trees (d) fresh plough-land (e) groundwork before building over. Note that the ground truth does NOT contain change classification, only binary change-no change decision for each pixel. For more information please contact Csaba Benedek.

## Terms of usage
The benchmark set is free for scientific use.
- Please acknowledge the use of the benchmark by referring to their relevant publications [1] and [2].
- Please notify us if a publication using the benchmark set appears.

## References
[1] Cs. Benedek and T. Szirányi: ”Change Detection in Optical Aerial Images by a Multi-Layer Conditional Mixed Markov Model”, IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 10, pp. 3416-3430, 2009
[2] Cs. Benedek and T. Szirányi: ”A Mixed Markov Model for Change Detection in Aerial Photos with Large Time Differences”, International Conference on Pattern Recognition (ICPR), Tampa, Florida, USA, December 8-11, 2008