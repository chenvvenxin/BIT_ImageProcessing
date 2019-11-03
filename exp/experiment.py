import PIL
import cv2
import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
from skimage.color.rgb_colors import red

from Camshift import start_Cam
import numpy as np
import matplotlib.pyplot as plt
from svm import svm
from tkinter import *

kernel_Laplace = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]])


def ReadImage(n):
    img = cv2.imread("lean.png")
    if n == 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ShowImage(img):
    cv2.imshow("Image", img)
    # cv2.waitKey(0)


def origin():
    img = ReadImage(1)
    ShowImage(img)


def gray():
    img = ReadImage(0)
    ShowImage(img)


def GrayscaleHistogram():
    img = ReadImage(0)
    plt.hist(img.ravel(), 256)
    plt.show()


def EqualizeHistImageC():
    img = ReadImage(1)
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    ShowImage(result)


def EqualizeHistImage():
    img = ReadImage(0)
    dst = cv2.equalizeHist(img)
    ShowImage(dst)


def LaplaceConvolution():
    img = ReadImage(0)
    output = cv2.filter2D(img, -1, kernel_Laplace)
    ShowImage(output)


def CannyThreshold():
    img = ReadImage(0)
    detected_edges = cv2.blur(img, (3, 3))
    detected_edges = cv2.Canny(detected_edges, 0, 0, apertureSize=3)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    ShowImage(dst)


def SobelThreshold():
    img = ReadImage(0)
    sobelXY = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    ShowImage(sobelXY)


def LaplaceThreshold():
    img = ReadImage(0)
    lap = cv2.Laplacian(img, cv2.CV_64F)  # 拉普拉斯边缘检测
    lap = np.uint8(np.absolute(lap))
    ShowImage(lap)


def MeanFilter():
    img = ReadImage(0)
    res = cv2.blur(img, ksize=(5, 5))
    ShowImage(res)


def MedianFilter():
    img = ReadImage(0)
    output = cv2.medianBlur(img, 5)
    ShowImage(output)


def GaussianBlur():
    img = ReadImage(0)
    output = cv2.GaussianBlur(img, (5, 5), 0)
    ShowImage(output)


def threshold():
    pic = ReadImage(0)
    ret, output = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    ShowImage(output)


def otsu():
    pic = ReadImage(0)
    ret, th = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)
    ShowImage(th)


def template_match():
    target = cv2.imread("lean.png")
    tpl = cv2.imread("nose.png")
    # methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    result = cv2.matchTemplate(target, tpl, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tl = min_loc
    br = (tl[0] + tw, tl[1] + th)
    cv2.rectangle(target, tl, br, [0, 255, 0], 2)
    cv2.imshow("match", target)


def affine_transformation():
    img = ReadImage(1)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows + 10))
    ShowImage(dst)


def perspective_transformation():
    pic = ReadImage(1)
    w, h = pic.shape[0:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    pts1 = np.float32([[20, 20], [200, h - 36], [w - 10, h - 10], [w - 60, 60]])
    M = cv2.getPerspectiveTransform(pts, pts1)
    dst = cv2.warpPerspective(pic, M, (600, 600))
    ShowImage(dst)


def Gradient_sharpening():
    pic = ReadImage(0)
    output = pic.copy()
    h, w = pic.shape[:2]
    for i in range(h - 1):
        for j in range(w - 1):
            output[i][j] = max(abs(pic[i][j + 1] - pic[i][j]), abs(pic[i + 1][j] - pic[i][j]))
    ShowImage(output)


def openning():
    pic = ReadImage(0)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel)
    ShowImage(opening)


def Feature_point_matching():
    MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
    template = cv2.imread('eyes.png', 0)  # queryImage
    target = ReadImage(0)  # trainImage
    # Initiate SIFT detector创建sift检测器
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    plt.imshow(result, 'gray')
    plt.show()


top = tk.Tk()
top.title('Image Processing')
# top.geometry('900x600')  # 窗口尺寸
top = Canvas(top, width=890, height=600, bg='white')
top.pack()
top.create_line(190, 100, 190, 540, fill='red', dash=(4, 4), width=2)
top.create_line(530, 100, 530, 540, fill='red', dash=(4, 4), width=2)
top.create_line(720, 100, 720, 540, fill='red', dash=(4, 4), width=2)

ft = tkFont.Font(family='Fixdsys', size=22, weight=tkFont.BOLD, slant=tkFont.ITALIC)
# Button控件
tk.Button(top, text='读取原图片', width=13, height=1, command=origin, bg="yellow").place(x=50, y=120)
tk.Button(top, text='灰度化', width=13, height=1, command=gray, bg="yellow").place(x=50, y=180)
tk.Button(top, text='灰度直方图', width=13, height=1, command=GrayscaleHistogram).place(x=50, y=240)
tk.Button(top, text='彩色直方图均衡化', width=13, height=1, command=EqualizeHistImageC).place(x=50, y=300)
tk.Button(top, text='灰度直方图均衡化', width=13, height=1, command=EqualizeHistImage).place(x=50, y=360)

tk.Button(top, text='阈值分割', width=13, height=1, command=threshold).place(x=220, y=120)
tk.Button(top, text='梯度锐化', width=13, height=1, command=Gradient_sharpening, bg="yellow").place(x=220, y=180)
tk.Button(top, text='Laplace锐化', width=13, height=1, command=LaplaceConvolution, bg="yellow").place(x=220, y=240)
tk.Button(top, text='Sobel边缘检测', width=13, height=1, command=SobelThreshold, bg="yellow").place(x=220, y=300)
tk.Button(top, text='Laplace边缘检测', width=13, height=1, command=LaplaceThreshold).place(x=220, y=360)
tk.Button(top, text='Canny算子', width=13, height=1, command=CannyThreshold).place(x=220, y=420)
tk.Button(top, text='均值滤波', width=13, height=1, command=MeanFilter, bg="yellow").place(x=220, y=480)
tk.Button(top, text='中值滤波', width=13, height=1, command=MedianFilter, bg="yellow").place(x=390, y=120)
tk.Button(top, text='高斯滤波', width=13, height=1, command=GaussianBlur, bg="yellow").place(x=390, y=180)
tk.Button(top, text='形态学滤波（开）', width=13, height=1, command=openning, bg="yellow").place(x=390, y=240)
tk.Button(top, text='仿射变换', width=13, height=1, command=affine_transformation, bg="yellow").place(x=390, y=300)
tk.Button(top, text='透视变换', width=13, height=1, command=perspective_transformation, bg="yellow").place(x=390, y=360)

tk.Button(top, text='otsu自适应阈值分割', width=17, height=1, command=otsu, bg="yellow").place(x=560, y=120)
tk.Button(top, text='模板匹配', width=13, height=1, command=template_match, bg="yellow").place(x=560, y=180)
tk.Button(top, text='特征点匹配', width=13, height=1, command=Feature_point_matching, bg="yellow").place(x=560, y=240)
tk.Button(top, text='SVM', width=13, height=1, command=svm, bg="yellow").place(x=560, y=300)

tk.Button(top, text='Camshift算法', width=13, height=1, command=start_Cam, bg="yellow").place(x=750, y=120)

im = PIL.Image.open("BITlogo.jpg")
img = ImageTk.PhotoImage(im)
imLabel = tk.Label(top, image=img, bd=0).place(x=350, y=13)
tk.Label(top, text='作者：陈文欣', bg='white').place(x=720, y=540)
tk.Label(top, text='邮箱：chenvvenxin@126.com', bg='white').place(x=720, y=560)
top.mainloop()
cv2.destroyAllWindows()
