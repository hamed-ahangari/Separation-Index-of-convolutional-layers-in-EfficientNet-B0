
# Separation Index of convolutional layers in EfficientNet-B0
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TjVkqSxdXMm8z5B_pBXMoo9tT5gbFeCa?usp=sharing)

In [this Jupyter Notebook](https://github.com/hamed-ahangari/Separation-Index-of-convolutional-layers-in-EfficientNet-B0/blob/master/SI_in_EfficientNetB0.ipynb), I've tried to calculate the Separation Index of the convolutional layers' outputs in a pretrained EfficientNet-B0.

## Separation Index
Separation and Smoothness indices are explaind in [this paper](https://arxiv.org/abs/1906.05156); Short from the abstract:

`
For classification problems, the separation rate of target labels in the space of dataflow is explained as a key factor indicating the performance of designed layers in improving the generalization of the network. According to the explained concept, a shapeless distance‐based evaluation index is proposed. Similarly, for regression problems, the smoothness rate of target outputs in the space of dataflow is explained as a key factor indicating the performance of designed layers in improving the generalization of the network. According to the explained smoothness concept, a shapeless distance‐based smoothness index is proposed for regression problems. To consider more strictly concepts of separation and smoothness, their extended versions are introduced, and by interpreting a regression problem as a classification problem, it is shown that the separation and smoothness indices are related together.
`

## Code Specifications
- **Deep learning library**: [Tensorflow 2](https://www.tensorflow.org/tutorials/quickstart/beginner) & [Keras API](https://www.tensorflow.org/guide/keras/functional)
- **Environment**: [Google Colab](https://colab.research.google.com/)
- **Model**: [EfficientNet - B0](https://arxiv.org/abs/1905.11946)
- **Pretrained Weights**: [NoisyStudent](https://arxiv.org/abs/1911.04252).
- **Fine-tunning approach**: 25 epochs with trainable dense layer, then 10 epochs with few unfreezed layers at the end of the model
- **Dataset**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Separation Index concept and formulas**: [Evaluation of Dataflow through layers of Deep Neural Networks in Classification and Regression Problems](https://arxiv.org/ftp/arxiv/papers/1906/1906.05156.pdf)

## Plots
- SI values of some of the layers in convolutional blocks through the model![## **SI values of some of the layers**](https://raw.githubusercontent.com/hamed-ahangari/Separation-Index-of-convolutional-layers-in-EfficientNet-B0/master/Results/SI%20values%20of%20some%20layers%20in%20convolutional%20blocks%20through%20the%20model.png)

- Number of parameters of the convolutional layers
![Number of the convolutional layers' parameters](https://raw.githubusercontent.com/hamed-ahangari/Separation-Index-of-convolutional-layers-in-EfficientNet-B0/master/Results/Number%20of%20parameters%20of%20the%20convolutional%20layers.png)

- Depths of the convolutional layers
![Depths of the convolutional layers](https://raw.githubusercontent.com/hamed-ahangari/Separation-Index-of-convolutional-layers-in-EfficientNet-B0/master/Results/Depths%20of%20the%20convolutional%20layers.png)

## References
- Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International Conference on Machine Learning. PMLR, 2019.: [paper](https://arxiv.org/abs/1905.11946)
- Kalhor, Ahmad, et al. "Evaluation of Dataflow through layers of Deep Neural Networks in Classification and Regression Problems." _arXiv preprint arXiv:1906.05156_ (2019).: [paper](https://arxiv.org/abs/1906.05156)+[code](https://github.com/melika-kheirieh/Seprability-index-CNN)
- Image classification via fine-tuning with EfficientNet: [blog post](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
- The official EfficientNet model repository: [GitHub repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- Stanford Dogs Dataset: [home page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Extra
You could use following links to know more about the EfficientNet architecture.
- [EfficientNet: Improving Accuracy and Efficiency through AutoML and Model Scaling](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html "Google AI Blog post about EfficientNet ")
- [Complete Architectural Details of all EfficientNet Models](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142 "Complete Architectural Details of all EfficientNet Models")
- [Image Classification with EfficientNet: Better performance with computational efficiency](https://medium.com/analytics-vidhya/image-classification-with-efficientnet-better-performance-with-computational-efficiency-f480fdb00ac6 "Image Classification with EfficientNet: Better performance with computational efficiency")
