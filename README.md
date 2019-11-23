# Deep Learning for Computer Vision
Computer Vision has become ubiquitous in our society, with applications in image/video search and understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, segmentation, localization and detection. Recent developments in neural network (a.k.a. deep learning) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems.

## PART1. Principal Component Analysis and k-Nearest Neighbors Classification

### Usage
Download the dataset you want to apply PCA to, and use the following command:
```
    cd PCA&KNN/
    python3 ./pca.py
```
Remenber that you dataset should be located in "./PCA&KNN/train" as well as "./PCA&KNN/test"

### Results
1. Perform PCA on the training set. Plot the mean face and the first three eigenfaces.


<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/mean_face.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_1.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_2.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_3.PNG" width="75%" height="75%">

2. Plot these reconstructed images using the first n = 3, 50, 100, 239 eigenfaces, with the corresponding MSE values.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig3.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig50.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig100.PNG" width="75%" height="75%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig239.PNG" width="75%" height="75%">

3. To apply the k-nearest neighbors classifier to recognize test set images, and use such hyperparameters, k = {1, 3, 5} and n = {3, 50, 159}. Show the 3-fold cross validation results.

![KNN-results](https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/KNN-results.png)

## PART2. Segmentation

### Usage
* filterBank.mat: The given .mat file contains a set of 38 filters (also known as filter
bank). This filter bank is stored as a 49 x 49 x 38 matrix (i.e., each filter is of size 49 x
49 pixels).
* Images: zebra.jpg and mountain.jpg
```
    cd Segmentation/
    python3 ./color_seg.py
    python3 ./text_seg.py
```
### result
1. Original images

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/mountain.jpg" width="50%" height="50%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/zebra.jpg" width="50%" height="50%">

2. Color segmentation

Convert both RGB images into Lab color space and plot the segmentation results for both images based on the clustering results

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/color_segmentation/Mountain.PNG" width="50%" height="50%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/color_segmentation/Zebra.PNG" width="50%" height="50%">


3. Texture segmentation

Convert the color images into grayscale ones, before extracting image textural features via the provided filter bank and plot the texture segmentation results for both images.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/texture_segmentation/Mountain.jpg" width="50%" height="50%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/texture_segmentation/Zebra.jpg" width="50%" height="50%">

4. Combine both color and texture features (3 + 38 = 41-dimensional features) for
image segmentation

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/combined_segmentation/Mountain.jpg" width="50%" height="50%">
<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/combined_segmentation/Zebra.jpg" width="50%" height="50%">
