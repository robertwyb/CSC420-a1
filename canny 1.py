import cv2
import numpy as np
from matplotlib import pyplot as plt


def correlation(image, patch, mode):
    img = cv2.imread(image)
    mid = int((patch.shape[0] - 1) / 2)
    i_shape, p_shape = img.shape, patch.shape     # get image amd patch size
    img_h, img_w = i_shape[0], i_shape[1]
    patch_h, patch_w = p_shape[0], p_shape[1]
    p_size = 2 * mid

    if mode == 'same':                            # based on input type create corresponding pad
        output = np.zeros((img_h, img_w, 3))
        pseudo = np.zeros((img_h + p_size, img_w + p_size, 3))
        pseudo[mid:mid+img_h, mid:mid+img_w] = img
    elif mode == 'valid':
        output = np.zeros((img_h - p_size, img_w - p_size, 3))
        pseudo = np.zeros((img_h, img_w, 3))
        pseudo = img
    elif mode == 'full':
        output = np.zeros((img_h + p_size, img_w + p_size, 3))
        pseudo = np.zeros((img_h + 2*p_size, img_w + 2*p_size, 3))
        pseudo[p_size:p_size+img_h, p_size:p_size+img_w] = img

    for row in range(output.shape[0]):           # multiply the filter with the image
        for col in range(output.shape[1]):
            square = np.array(pseudo[row:row+patch_h, col:col+patch_w])
            r_sq = np.reshape(square, (-1, 3))
            r_patch = np.reshape(patch, -1)
            res = np.zeros(3)
            for i in range(r_sq.shape[0]):
                res += r_sq[i] * r_patch[i]
            output[row, col] = res

    return output


def gaussian_kernel(sigmax, sigmay):             # use cv2.getGaussianKernel to get required filter
    gausx = cv2.getGaussianKernel(5, sigmax)
    gausy = cv2.getGaussianKernel(5, sigmay)
    gausy = np.reshape(gausy, (1, -1))
    kernel = np.multiply(gausx, gausy)
    return kernel

def gaussian_filter(image, mode):
    filter = gaussian_kernel(3, 5)
    print(filter)
    filter = np.flipud(filter)                   # flip the filter to perform convolution
    filter = np.fliplr(filter)
    output = correlation(image, filter, mode)
    return output


# output = gaussian_filter('iris.jpg', 'same')
# cv2.imwrite('result.jpg', output)
# cv2.imshow("output", output)
# cv2.waitKey(0)








#
window_name = "Laplace Demo"
window_name1 = "Horizontal derivative Demo"
window_name2 = "Vertical derivative Demo"
img = cv2.imread('portrait.jpg')
img = cv2.GaussianBlur(img, (3, 3), 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
derx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, 3)
dery = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, 3)
derxy = cv2.Sobel(derx, cv2.CV_64F, 0, 1, 3)
lap = cv2.Laplacian(img_gray, cv2.CV_64F, 1)
# cv2.imwrite('derx.jpg', derx)
# cv2.imwrite('dery.jpg', dery)
# cv2.imwrite('derxy.jpg', derxy)
cv2.imwrite('lap1.jpg', lap)

#
# waldo = cv2.imread('waldo.jpg')
# wshape = waldo.shape
# scene = cv2.imread('whereswaldo.jpg')
# res = cv2.matchTemplate(scene, waldo, cv2.TM_CCOEFF)
# minmax = cv2.minMaxLoc(res)
# min_val, max_val, min_loc, max_loc = minmax
# corner1 = max_loc
# corner2 = (max_loc[0] + wshape[1], max_loc[1] + wshape[0] )
# cv2.rectangle(scene, corner1, corner2, (255, 0, 0), 3)
# cv2.imshow("output", scene)
# cv2.waitKey(0)
#
#
#
img = cv2.imread('portrait.jpg',0)
edges = cv2.Canny(img,120,470)
cv2.imwrite('canny.jpg', edges)
cv2.imshow("output", edges)
cv2.waitKey(0)
