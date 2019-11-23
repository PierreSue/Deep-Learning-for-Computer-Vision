# Deep Learning for Computer Vision
Computer Vision has become ubiquitous in our society, with applications in image/video search and understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, segmentation, localization and detection. Recent developments in neural network (a.k.a. deep learning) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems.

## PART1. Principal Component Analysis and k-Nearest Neighbors Classification

### Usage
Download the dataset you want to apply PCA to, and use the following command:
'''
    cd PCA&KNN/
    python3 ./pca.py
'''
Remenber that you dataset should be located in "./PCA&KNN/train" as well as "./PCA&KNN/test"

### Results
1. Perform PCA on the training set. Plot the mean face and the first three eigenfaces.
![meanface](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/mean_face.PNG)
![eigenface1](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_1.PNG)

![eigenface2](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_2.PNG)
![eigenface3](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_3.PNG)

2. Plot these reconstructed images using the first n = 3, 50, 100, 239 eigenfaces, with the corresponding MSE values.
![eigen3](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig3.PNG)
![eigen50](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig50.PNG)

![eigen100](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig100.PNG)
![eigen239](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig239.PNG)

3. To apply the k-nearest neighbors classifier to recognize test set images, and use such hyperparameters, k = {1, 3, 5} and n = {3, 50, 159}. Show the 3-fold cross validation results.

![eigen239](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/KNN-results.PNG)