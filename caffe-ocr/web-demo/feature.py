import time
import math
import logging
import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
coef_cni_nom = {'alpha': 0.34, 'beta': 2.5, 'ratio': 5, 'hratio': 1.1}
coef_cni_prenom = {'alpha': 0.38, 'beta': 4.3, 'ratio': 9, 'hratio': 1.1}
coef_cni_naissance = {'alpha': 0.3, 'beta': 6.8, 'ratio': 6, 'hratio': 1.17}

coef_passport_nom = {'alpha': 0, 'beta': 3.6, 'ratio': 5, 'hratio': 1.2}
coef_passport_prenom = {'alpha': 0, 'beta': 5.4, 'ratio': 8, 'hratio': 1.2}
coef_passport_naissance = {'alpha': 0.45, 'beta': 8.8, 'ratio': 10, 'hratio': 1.3}

coefs = [(coef_cni_nom, coef_cni_prenom, coef_cni_naissance), (coef_passport_nom, coef_passport_prenom, coef_passport_naissance)]

def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def isRectangle(pts, seuil=0.05):
    if len(pts) != 4:
        raise ValueError('Input must be 4 points')
    coords = pts.ravel().reshape(-1, 2)
    cx, cy = np.mean([coord[0] for coord in coords]), np.mean(
        [coord[1] for coord in coords])
    dist = [distance((cx, cy), coord) for coord in coords]
    res = 0
    # print coords
    for i in xrange(1, 4):
        res += abs(dist[0] - dist[i])
    bias = res / distance(coords[1], coords[2])
    logging.info('Regtangle bias: %.3f', res)
    logging.info('Ratio bias: %.3f', bias)
    if bias < seuil:
        line1 = coords[3] - coords[0]
        line2 = coords[2] - coords[1]
        mean_radian = - \
            (math.atan2(line1[1], line1[0]) +
             math.atan2(line2[1], line2[0])) / 2
        inclination = math.degrees(mean_radian)  # / np.pi * 90
        logging.info('Document rotation: %.3f degree', inclination)
        return True, mean_radian
    else:
        return False, None

def rotateImage(image, angle):
    if abs(angle) > 60:
        return image
    row, col = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image

def convert_ndarray(pts, w=2, h=4):
    pts_list = pts.ravel().tolist()
    list_2d = [[pts_list[w * y + x] for x in range(w)] for y in range(h)]
    return list_2d

def find_zone(corners, theta, alpha, beta, ratio, hratio):  # , theta, alpha, beta
    # x = x0 + alpha * delta(x) + beta * height * sin(theta)
    # y = y0 + alpha * delta(x) * sin(theta) - beta * height * cos(theta)
    delta_x = corners[3][0] - corners[0][0]
    height = (math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
              + math.hypot(corners[2][0] - corners[3][0], corners[2][1] - corners[3][1])) / 2
    # height = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    x0, y0 = corners[0]
    h = hratio * height
    w = ratio * h
    adjust_x = (0, h * math.sin(theta))[theta < 0]  # (false, true)[cond]
    adjust_y = (0, w * math.sin(theta))[theta > 0]

    tfx = x0 + alpha * delta_x + beta * height * math.sin(theta)
    tfy = y0 - alpha * delta_x * \
        math.sin(theta) + beta * height * math.cos(theta)
    top_left = (int(tfx + adjust_x), int(tfy - adjust_y))

    brx = tfx + w * math.cos(theta) + h * math.sin(theta)
    bry = tfy + h * math.cos(theta) - w * math.sin(theta)
    bottom_right = (int(brx - adjust_x), int(bry + adjust_y))

    return (top_left, bottom_right)

def feature_matching(queryImage, trainImage, doctype=0):
    img_nom, img_prenom, img_naissance = None, None, None

    img1 = cv2.imread(queryImage, 0)
    if isinstance(trainImage, str):
        img2_color = cv2.imread(trainImage)
    else:
        img2_color = trainImage
    print img2_color.shape
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # k best matches found per each query descriptor
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    isrect, theta = False, 0.0
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return None, img_nom, img_prenom, img_naissance
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        corners = convert_ndarray(dst)
        isrect, theta = isRectangle(dst)

        if isrect:
            logging.info('CNI found')
            inclination = - math.degrees(theta)
            # print inclination
            # draw nom and prenom
            coef_nom, coef_prenom, coef_naissance =  coefs[doctype]
            tl, br = find_zone(corners, theta, **coef_nom)
            cv2.rectangle(img2_color, tl, br, (255, 0, 0), 3)
            img_nom = img2[tl[1]:br[1], tl[0]:br[0]]
            img_nom = rotateImage(img_nom, inclination)

            tl, br = find_zone(corners, theta, **coef_prenom)
            cv2.rectangle(img2_color, tl, br, (255, 0, 0), 3)
            img_prenom = img2[tl[1]:br[1], tl[0]:br[0]]
            img_prenom = rotateImage(img_prenom, inclination)

            tl, br = find_zone(corners, theta, **coef_naissance)
            cv2.rectangle(img2_color, tl, br, (255, 0, 0), 3)
            img_naissance = img2[tl[1]:br[1], tl[0]:br[0]]
            img_naissance = rotateImage(img_naissance, inclination)

        img2_color = cv2.polylines(
            img2_color, [np.int32(dst)], True, (0, 0, 255), 4, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2_color, kp2,
                           good, None, **draw_params)
    return (isrect, math.degrees(theta)), img3, img_nom, img_prenom, img_naissance


CNI_PATH = 'images/cni/'
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    queryImage = CNI_PATH + 'motif/cniNB.png'
    # trainImage = 'images/cni/' + 'CNI_femme_paysage_NB_2.tif'
    # trainImage = 'images/cni/' + 'CNI_femme_paysage_NB.tif'
    # trainImage = CNI_PATH + 'converted/CNI_double_paysage_NB.pdf-0.png'
    trainImage = CNI_PATH + 'converted/CNI_femme_paysage_couleur_prenom_compose.pdf-0.png'

    start_time = time.time()
    (isrect, theta), img, img_nom, img_prenom, img_naissance = feature_matching(queryImage, trainImage)
    end_time = time.time()

    logging.info("Processing time: %.3f seconds" % (end_time - start_time))
    plt.imshow(img, 'gray'), plt.show()
