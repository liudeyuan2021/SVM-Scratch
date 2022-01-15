##
## Created by Liu Deyuan on 2021/01/15.
##

import time
import numpy as np


def cell_hog(magnitude, orientation, orientation_start, orientation_end,
             cell_columns, cell_rows, column_index, row_index,
             size_columns, size_rows, range_rows_start, range_rows_stop,
             range_columns_start, range_columns_stop):
    """Calculation of the cell's HOG value
    Parameters
    ----------
    magnitude : ndarray
        The gradient magnitudes of the pixels.
    orientation : ndarray
        Lookup table for orientations.
    orientation_start : float
        Orientation range start.
    orientation_end : float
        Orientation range end.
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    column_index : int
        Block column index.
    row_index : int
        Block row index.
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    range_rows_start : int
        Start row of cell.
    range_rows_stop : int
        Stop row of cell.
    range_columns_start : int
        Start column of cell.
    range_columns_stop : int
        Stop column of cell
    Returns
    -------
    total : float
        The total HOG value.
    """
    total = 0.

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


def hog_histograms(gradient_columns, gradient_rows, cell_columns, cell_rows,
                   size_columns, size_rows, number_of_cells_columns, number_of_cells_rows,
                   number_of_orientations, orientation_histogram):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.
    Parameters
    ----------
    gradient_columns : ndarray
        First order image gradients (rows).
    gradient_rows : ndarray
        First order image gradients (columns).
    cell_columns : int
        Pixels per cell (rows).
    cell_rows : int
        Pixels per cell (columns).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    number_of_cells_columns : int
        Number of cells (rows).
    number_of_cells_rows : int
        Number of cells (columns).
    number_of_orientations : int
        Number of orientation bins.
    orientation_histogram : ndarray
        The histogram array which is modified in place.
    """

    magnitude = np.hypot(gradient_columns, gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180

    r_0 = cell_rows // 2
    c_0 = cell_columns // 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    range_rows_stop = (cell_rows + 1) // 2
    range_rows_start = -(cell_rows // 2)
    range_columns_stop = (cell_columns + 1) // 2
    range_columns_start = -(cell_columns // 2)
    number_of_orientations_per_180 = 180. // number_of_orientations

    # compute orientations integral images
    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                orientation_histogram[r_i, c_i, i] = \
                        cell_hog(magnitude, orientation,
                                        orientation_start, orientation_end,
                                        cell_columns, cell_rows, c, r,
                                        size_columns, size_rows,
                                        range_rows_start, range_rows_stop,
                                        range_columns_start, range_columns_stop)
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows

def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out


def _hog_channel_gradient(channel):
    """Compute unnormalized gradient image along `row` and `col` axes.

    Parameters
    ----------
    channel : (M, N) ndarray
        Grayscale image or one of image channel.

    Returns
    -------
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """
    g_row = np.empty(channel.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col

def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        block_norm='L2-Hys', visualize=False, transform_sqrt=False,
        feature_vector=True, multichannel=None):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

        1. (optional) global image normalization
        2. computing the gradient image in `row` and `col`
        3. computing gradient histograms
        4. normalizing across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    image : (M, N[, C]) ndarray
        Input image.
    orientations : int, optional
        Number of orientation bins.
    pixels_per_cell : 2-tuple (int, int), optional
        Size (in pixels) of a cell.
    cells_per_block : 2-tuple (int, int), optional
        Number of cells in each block.
    block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}, optional
        Block normalization method:

        ``L1``
           Normalization using L1-norm.
        ``L1-sqrt``
           Normalization using L1-norm, followed by square root.
        ``L2``
           Normalization using L2-norm.
        ``L2-Hys``
           Normalization using L2-norm, followed by limiting the
           maximum values to 0.2 (`Hys` stands for `hysteresis`) and
           renormalization using L2-norm. (default)
           For details, see [3]_, [4]_.

    visualize : bool, optional
        Also return an image of the HOG.  For each cell and orientation bin,
        the image contains a line segment that is centered at the cell center,
        is perpendicular to the midpoint of the range of angles spanned by the
        orientation bin, and has intensity proportional to the corresponding
        histogram value.
    transform_sqrt : bool, optional
        Apply power law compression to normalize the image before
        processing. DO NOT use this if the image contains negative
        values. Also see `notes` section below.
    feature_vector : bool, optional
        Return the data as a feature vector by calling .ravel() on the result
        just before returning.
    multichannel : boolean, optional
        If True, the last `image` dimension is considered as a color channel,
        otherwise as spatial.

    Returns
    -------
    out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray
        HOG descriptor for the image. If `feature_vector` is True, a 1D
        (flattened) array is returned.
    hog_image : (M, N) ndarray, optional
        A visualisation of the HOG image. Only provided if `visualize` is True.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    .. [2] Dalal, N and Triggs, B, Histograms of Oriented Gradients for
           Human Detection, IEEE Computer Society Conference on Computer
           Vision and Pattern Recognition 2005 San Diego, CA, USA,
           https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf,
           :DOI:`10.1109/CVPR.2005.177`

    .. [3] Lowe, D.G., Distinctive image features from scale-invatiant
           keypoints, International Journal of Computer Vision (2004) 60: 91,
           http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf,
           :DOI:`10.1023/B:VISI.0000029664.99615.94`

    .. [4] Dalal, N, Finding People in Images and Videos,
           Human-Computer Interaction [cs.HC], Institut National Polytechnique
           de Grenoble - INPG, 2006,
           https://tel.archives-ouvertes.fr/tel-00390303/file/NavneetDalalThesis.pdf

    Notes
    -----
    The presented code implements the HOG extraction method from [2]_ with
    the following changes: (I) blocks of (3, 3) cells are used ((2, 2) in the
    paper); (II) no smoothing within cells (Gaussian spatial window with sigma=8pix
    in the paper); (III) L1 block normalization is used (L2-Hys in the paper).

    Power law compression, also known as Gamma correction, is used to reduce
    the effects of shadowing and illumination variations. The compression makes
    the dark regions lighter. When the kwarg `transform_sqrt` is set to
    ``True``, the function computes the square root of each color channel
    and then applies the hog algorithm to the image.
    """
    image = np.atleast_2d(image)

    if multichannel is None:
        multichannel = (image.ndim == 3)

    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with 2 spatial dimensions are '
                         'supported. If using with color/multichannel '
                         'images, specify `multichannel=True`.')

    """
    The first stage applies an optional global image normalization
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each color channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if transform_sqrt:
        image = np.sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    color channel is used, which provides color invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if image.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers
        image = image.astype('float')

    # 本项目中使用的是单通道的图片，如果C++实现中碰到问题可去掉多通道的情况
    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype=np.double)
        g_col_by_ch = np.empty_like(image, dtype=np.double)
        g_magn = np.empty_like(image, dtype=np.double)

        for idx_ch in range(image.shape[2]):
            g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
                _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
                                            g_col_by_ch[:, :, idx_ch])

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=2)
        rr, cc = np.meshgrid(np.arange(image.shape[0]),
                             np.arange(image.shape[1]),
                             indexing='ij',
                             sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        g_row, g_col = _hog_channel_gradient(image)

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))

    hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                                 n_cells_col, n_cells_row,
                                 orientations, orientation_histogram)

    # 如果没有visualize的需求，C++的实现可去掉这一部分
    # now compute the histogram for each cell
    hog_image = None

    if visualize:
        from .. import draw

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = draw.line(int(centre[0] - dc),
                                       int(centre[1] + dr),
                                       int(centre[0] + dc),
                                       int(centre[1] - dr))
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

    """
    The fourth stage computes normalization, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalization introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalizations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalizations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog_normalize_block(block, method=block_norm)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()

    if visualize:
        return normalized_blocks, hog_image
    else:
        return normalized_blocks

def feature_extraction(arr):
    """ Return a list of feature vectors of the arr, each image is arr[i] """
    arr_feature = []
    for i in range(arr.shape[0]):
        arr_feature.append(hog(arr[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys'))
    return np.array(arr_feature)

if __name__ == '__main__':
    # (1)读取数据
    data = np.load('dataset/data_float16.npz')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    # (2)提取数据特征
    print('Begin Feature Extraction')
    X_train_feature = feature_extraction(X_train[:10])
    start_time = time.time()
    X_test_feature = feature_extraction(X_test)
    end_time = time.time()
    print("{:f}s for {:d} test feature extraction".format(end_time - start_time, y_test.shape[0]))
    print()

    print(X_train_feature.shape)
    print(X_test_feature.shape)