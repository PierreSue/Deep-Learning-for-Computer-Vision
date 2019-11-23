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


<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/mean_face.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_1.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_2.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/eigenface/eigenface_3.PNG" width="20%" height="20%">

2. Plot these reconstructed images using the first n = 3, 50, 100, 239 eigenfaces, with the corresponding MSE values.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig3.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig50.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig100.PNG" width="20%" height="20%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/reconstruction/eig239.PNG" width="20%" height="20%">

3. To apply the k-nearest neighbors classifier to recognize test set images, and use such hyperparameters, k = {1, 3, 5} and n = {3, 50, 159}. Show the 3-fold cross validation results.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/PCA%26KNN/KNN-results.png" width="70%" height="70%">

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
### Results
1. Original images

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/mountain.jpg" width="30%" height="30%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/zebra.jpg" width="30%" height="30%">

2. Color segmentation

Convert both RGB images into Lab color space and plot the segmentation results for both images based on the clustering results

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/color_segmentation/Mountain.PNG" width="30%" height="30%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/color_segmentation/Zebra.PNG" width="30%" height="30%">


3. Texture segmentation

Convert the color images into grayscale ones, before extracting image textural features via the provided filter bank and plot the texture segmentation results for both images.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/texture_segmentation/Mountain.jpg" width="30%" height="30%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/texture_segmentation/Zebra.jpg" width="30%" height="30%">

4. Combine both color and texture features (3 + 38 = 41-dimensional features) for
image segmentation

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/combined_segmentation/Mountain.jpg" width="30%" height="30%"> <img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Segmentation/combined_segmentation/Zebra.jpg" width="30%" height="30%">


## PART3. Recognition with Bag of Visual Words

### Usage
The implementation of a basic image-based bag-of-words (BoW) model for a scene image dataset with 5 categories.
```
    cd Recognition/
    python3 ./interst_point_detection.py
    python3 ./kmeans.py
```

### Results
1. Detect interest points, calculate their descriptors for this image using SURF, and plot the interest point detection results

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Recognition/image/interest_point_detection.jpg" width="60%" height="60%">

2. Use k-means algorithm to divide these interest points into C clusters

Extract the detected interest points from all of the 50 images in Train-10, and stack them into a N × d matrix, where N denotes the total number of interest points and d is the dimension of its descriptor. (choose C = 50 and maximum number of iterations = 5000) The centroid of each cluster then indicates a visual word. Randomly select 6 clusters from the above results and plot the visual words and the associated interest points in this PCA subspace.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Recognition/image/kmeans.jpg" width="60%" height="60%">

## PART4. Semantic Segmentation

### Usage
Perform 2 semantic segmentation models, which predicts a label to each pixel with CNN models. The input is a RGB image while the output is the semantic segmentation/prediction. The models I used are the base model (VGG16-FCN32s) and the improved model(FCN8s). Before using the codes, the training/testing data and model file (vgg16_weights_tf_dim_ordering_tf_kernels.h5) are required.

```
cd Semantic-segmentation
bash ss.sh $1 $2
bash ss_best.sh $1 $2
    $1: testing images directory (images are named 'xxxx_sat.jpg')
    $2: output images directory
python3 mean_ios_evaluate.py -g ground_truth -p prediction
```
I also provide my pre-trained models for both. Check [model_files](https://www.dropbox.com/s/jcq61nhlbw12to0/b04901070_hw3.zip?dl=0) .

### Results
1. Show the predicted segmentation mask from the base model.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Semantic-segmentation/image/Base.png" width="60%" height="60%">


2. Show the predicted segmentation mask from the improved model.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Semantic-segmentation/image/Improved.png" width="60%" height="60%">

3. Demonstrate the IOU accuracy.

<img src="https://github.com/PierreSue/Deep-Learning-for-Computer-Vision/blob/master/Semantic-segmentation/image/IOU.png" width="60%" height="60%">