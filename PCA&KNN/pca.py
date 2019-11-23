import numpy as np
from sklearn.decomposition import PCA
import os
import scipy.misc

train_path = 'train/'
test_path = 'test/'

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

img_shape = scipy.misc.imread(train_path + train_files[0]).shape

train_faces = [] # 240
test_faces = [] # 160
for train_f in train_files:
    train_faces.append(np.array(scipy.misc.imread(train_path + train_f).flatten()))
for test_f in test_files:
    test_faces.append(np.array(scipy.misc.imread(test_path + test_f).flatten()))

#### mean face
mean_face = np.array(np.mean(train_faces, axis=0))
scipy.misc.imsave('mean.png', mean_face.reshape(img_shape))

### PCA
pca = PCA()
pca.fit(np.array(train_faces))
for i in range(3):
    scipy.misc.imsave('eigenface{}.png'.format(i), pca.components_[i].reshape(img_shape))

### recontruct 1_1.png with 3, 50, 100, 239 eigenfaces
img1_1 = scipy.misc.imread(train_path+'1_1.png').flatten()
weights = np.array([np.dot(img1_1-mean_face, pca.components_[i]) for i in range(len(train_files)-1)])
for i in [3, 50, 100, 239]:
    reconstuction = np.dot(weights[:i], pca.components_[:i])+mean_face
    scipy.misc.imsave('first{}.png'.format(i), reconstuction.reshape(img_shape))
    print('MSE for first{0} eigenface: {1}'.format(i, np.square(reconstuction-img1_1).mean()))

### k nearest neighbor
# k = 1, 3, 5
# n = 3, 50, 159
from sklearn.neighbors import KNeighborsClassifier
cros_val_fold = 3
part_size = int(len(train_files)/cros_val_fold)
for k in [1, 3, 5]:
    for n in [3, 50, 159]:
        score_list = []
        for i in range(cros_val_fold):
            neigh = KNeighborsClassifier(n_neighbors=k)
            part_idx = [0, 1, 2]
            subval_files = train_files[i*part_size:(i+1)*part_size]
            subval_label = [int(s[:s.find('_')]) for s in subval_files]
            subval_faces = train_faces[i*part_size:(i+1)*part_size]
            subval_data = [[np.dot(subval_face-mean_face, pca.components_[j]) for j in range(n)] for subval_face in subval_faces]
            del part_idx[i]
            subtrain_files = train_files[part_idx[0]*part_size:(part_idx[0]+1)*part_size]+train_files[part_idx[1]*part_size:(part_idx[1]+1)*part_size]
            subtrain_label = [int(s[:s.find('_')]) for s in subtrain_files]
            subtrain_faces = train_faces[part_idx[0]*part_size:(part_idx[0]+1)*part_size]+train_faces[part_idx[1]*part_size:(part_idx[1]+1)*part_size]
            subtrain_data = [[np.dot(subtrain_face-mean_face, pca.components_[j]) for j in range(n)] for subtrain_face in subtrain_faces]

            neigh.fit(subtrain_data, subtrain_label)
            # print('k={},n={},score={}'.format(k,n,neigh.score(subval_data, subval_label)))
            score_list.append(neigh.score(subval_data, subval_label))
        print('k={}, n={}, score={}'.format(k,n,np.mean(score_list)))

# k=1, n=159
best_k = 1
best_n = 159
best_neigh = KNeighborsClassifier(n_neighbors=best_k)
train_data = [[np.dot(face-mean_face, pca.components_[j]) for j in range(best_n)] for face in train_faces]
train_label = [int(s[:s.find('_')]) for s in train_files]
test_data = [[np.dot(face-mean_face, pca.components_[j]) for j in range(best_n)] for face in test_faces]
test_label = [int(s[:s.find('_')]) for s in test_files]

best_neigh.fit(train_data, train_label)
print('test score = {}'.format(best_neigh.score(test_data, test_label)))