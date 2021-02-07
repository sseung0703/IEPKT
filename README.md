# Interpretable Embedding Procedure Knowledge Transfer
- Implementation of "Interpretable embedding procedure knowledge transfer" on AAAI2021.

# Paper abstract
This paper proposes a method of generating interpretable embedding procedure knowledge based on principal component analysis, and distilling it based on a message passing neural network. Experimental results show that the student network trained by the proposed KD method improves 2.28% in the CIFAR100 dataset, which is higher performance than the state-of-the-art method. We also demonstrate that the embedding procedure knowledge is interpretable via visualization of the proposed KD process.

<p align="center">
<img width="400" alt="Conceptual diagram of the proposed method." src="https://user-images.githubusercontent.com/26036843/103818644-43fa3480-50ac-11eb-8140-a744588e2e3d.png">
</p>

## Requirements
- Tensorflow 1.x

## Visualization for interpreting the embedding procedure
- In order to show that our knowledge can interpret the embedding procedure, we visualize our knowledge.
- Below visualization results is coincide with human's understanding of how CNN operates.

Network: WResNet40-4
Dataset: CIFAR10 training set
1. 
In the former point in CNN, our knowledge shows that data is clustered based on low-level information, e.g., color and simple edges.
<p align="center">
  <img src="pics/video0.gif" width="400"><br>
  <b></b>
</p>

2.
In the middle point of CNN, our knowledge shows that embedding is on-going by more broadly distributed green data.
Note that, green data is the most common color in CIFAR10.
<p align="center">
  <img src="pics/video1.gif" width="400"><br>
  <b></b>
</p>

3.
In the last stage of CNN, dataset is well-clustered according to the classes.
<p align="center">
  <img src="pics/video2.gif" width="400"><br>
  <b></b>
</p>
