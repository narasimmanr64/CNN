import numpy as np

def padWithZeros(arr, rowCount, colCount):
    """
    Function to add zeros to the given array
    :param arr: Arr where zeros need to be added
    :param rowCount: Number of rows of zeros to be added
    :param colCount: Number of columns of zeros to be added
    :return: Expanded array
    """
    finalRowCount = arr.shape[0] + rowCount
    finalColCount = arr.shape[1] + colCount
    result = np.zeros((finalRowCount, finalColCount))
    result[0:arr.shape[0], 0:arr.shape[1]] = arr
    return result

def get_circulant(vec, number):
    """
    Function to get the (Row) circulant matrix of a given array
    :param vec: Vector whose circulant matrix need to be computed
    :param number: number of rows of the circulant matrix
    :return: circulant matrix
    """
    n = vec.shape[0]
    result = np.zeros((number, n))
    result[0, :] = vec
    for i in range(1, number):
        result[i, 1:n] = result[i-1,0:n-1]
        result[i, 0] = result[i-1,n-1]
    return result

def repeatRows(arr,number):
    """
    Function to repeat the number of rows in an array
    :param arr: Input array whose rows need to be repeated
    :param number: Number of times the rows need to be repeated
    :return: expanded array
    """
    result = np.zeros((arr.shape[0], arr.shape[1]*number))
    result[0:arr.shape[0],0:arr.shape[1]] = arr
    startIndex = 0
    for i in range(1, number):
        startIndex += arr.shape[1]
        endIndex = startIndex + arr.shape[1]-1
        result[0:arr.shape[0]-i,startIndex:endIndex+1] = arr[i:arr.shape[0],:]
    return result

def convolute(image, w, padOptions):
    """
    Function to calculate the convolution of an image and a weight vector.
    PadOptions will computer 'SAME' or reduced value
    :param image: Image matrix
    :param w: weight matrix
    :param padOptions: 'SAME' will pad the image with zeros so that the convolution is the same size as image
    :return: Convolution matrix
    """
    (m,n) = image.shape
    (p, q) = w.shape
    print(image.shape)
    print(w.shape)
    if padOptions == 'SAME':
        padImage = padWithZeros(image, p-1, q-1)
        (m, n) = padImage.shape
    else:
        padImage = image

    EpImage = repeatRows(padImage, p)
    padW = padWithZeros(w, 0, n-q)
    cfpadW = get_circulant(padW.flatten(), n-q+1)
    convolution = EpImage[0:m-p+1,:].dot(cfpadW.T)
    return convolution

X = np.random.randint(1,100,size=(1000,1000))
w = np.random.randint(1,10,size=(3,2))

convolution = convolute(X, w, 'VALID')
print(convolution)
