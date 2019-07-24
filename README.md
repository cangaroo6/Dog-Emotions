# Dog-Emotions
Training a model to predict dog emotions by facial expressions

This project had the goal to build a classifier, which can determine a dog's emotion by its facial expression. Therefore, we created an image library with dog faces showing different expressions that are recorded in the name of each picture. The model was built by training a convolutional neural network (CNN) on the image library. Two applications for the resulting model were build: A folder-based image classification tool and a webcam emotion classifier. 

Library: 400 images in total, 100 images per emotion (anger, fear, happiness, sadness)

Model Training:
- 10% test set, 90% training set
- 20% of training set for validation
- Epochs: 100
- Batch size: 16
- Accuracy: 45%-55%
