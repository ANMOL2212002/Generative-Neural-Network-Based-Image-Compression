# image compression using k-means clustering algorithm and k = 4

from skimage import io
from sklearn.cluster import KMeans
import numpy as np

def k_means_compression(image_path, k=4, save_path=None):
    image = io.imread(image_path)
    #Dimension of the original image
    rows = image.shape[0]
    cols = image.shape[1]
    d = image.shape[2]
    # print(image.shape)

    #Flatten the image
    image = image.reshape(rows*cols, d)

    #Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    #Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, d)

    #Save and display output image
    io.imsave(save_path, compressed_image)
    # io.imshow(compressed_image)
    # io.show()

if __name__ == '__main__':
    image = io.imread('test.png')
    k_means_compression(image, 2)

