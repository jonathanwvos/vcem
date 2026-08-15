import numpy as np


def gamma_adjustment(image, gamma, normal_value):
  '''
  Helper method to adjust the gamma of an image using the provided normal value.

  Arguments:
  image -- A grayscale or colour image as a numpy array
  gamma -- The gamma value by which to adjust the image. Values less than 1.0
            desaturate the image, values higher than 1.0 saturate the image
  '''

  if not isinstance(image, np.ndarray):
      raise TypeError('image must be a numpy array of an image')

  ndim = image.ndim

  if ndim not in [2,3]:
      raise TypeError('image must be either a grayscale image of dim=2 or a colour image with dim=3')

  if gamma < 0.0:
      raise ValueError('gamma must be a positive real number')

  return np.power(image / normal_value, gamma) * normal_value

def unique_pixels(image):
    '''
    Constructs a list of all unique pixels in the image. Typically used in image clustering.
    Accepts RGB, BGR and HSV colour formats.

    Arguments:
    image -- An image as a numpy array

    Returns list of unique pixels
    '''

    if type(image) is not np.ndarray:
        raise TypeError('Image must be a numpy array.')

    reference = np.zeros((256,256,256), dtype='bool')

    for v1,v2,v3 in image.reshape((image.shape[0]*image.shape[1],3)):
        reference[v1, v2, v3] = 1

    unique_colours = []

    for i in range(256):
        for j in range(256):
            for k in range(256):
                if reference[i,j,k] == 1:
                    unique_colours.append([i,j,k])

    return np.array(unique_colours, dtype='uint8')

def cluster_image_kmeans(image, no_clusters=10, return_centroids=False):
    '''
    Determine the colour centroids for an image and change all pixels to the closest one.

    Arguments:
    image       -- An image as a numpy array
    no_clusters -- The number of centroids to find

    Returns a copy of the image after the colours have been changed according the clustering.
    '''

    if type(image) is not np.ndarray:
        raise TypeError('Image must be a numpy array.')
    if type(no_clusters) is not int:
        raise TypeError('No. of clusters must be an integer.')
    if no_clusters < 1:
        raise ValueError('No. of clusters must be >1.')
    if image.ndim != 3:
        raise ValueError('Image must be a 3D array.')

    from sklearn.cluster import KMeans

    clt = KMeans(no_clusters)

    shape = (image.shape[0]*image.shape[1],3)

    clt.fit(image.reshape(shape))

    cluster_centers = np.rint(clt.cluster_centers_).astype('uint8')
    result = cluster_centers[clt.labels_.flatten()]
    result = result.reshape(image.shape)

    if return_centroids:
        return cluster_centers, result
    else:
        return result

def normalize_image(image):
    '''
    Helper method to normalize the pixel values of an image across all channels.

    Arguments:
    image --  A grayscale or colour image as a numpy array
    '''

    if not isinstance(image, np.ndarray):
        raise TypeError('im must be a numpy array of an image')

    image = image.astype('float32')/255.
    _max = np.max(image)
    _min = np.min(image)

    return (image - _min)/(_max - _min)
