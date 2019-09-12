
'''
Update on September 12, 2019
Digital Recognition
@author: zhangpeng bo (zhang26162@gmail.com)
'''
import cv2
import numpy as np
from keras.models import load_model


def changeDim(img):
    ''' 改变图片维度 '''

    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    return img


def DigitalRecognition(event, x, y, flags, param):
    ''' 鼠标绘图，并识别数字 '''

    global start, drawing, img_copy, flag

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)

    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # 是否为第一个数字
        if flag:
            tmp = (img_b - img) - (img_b - img_copy)
            img2 = img_b - tmp

            lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        else:
            flag = True
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        lab = lab[:,:,1]
        # 分割字符区域   
        maskOri = cv2.inRange(lab, threshold, 255) # 分割的中间结果图像
        contours, hierarchy = cv2.findContours(maskOri, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 识别字符
        newmask = img # 识别的最终结果图像
        for cnt in contours:
            # 获取矩形区域信息
            x, y, w, h = cv2.boundingRect(cnt)
            width = np.max([h, w])

            # 通过长宽做一次筛选
            if width > 30 and width < 150:            
                if h > w:
                    left = x + w//2 - h//2
                    up = y
                else:
                    left = x
                    up = y + h//2 - w//2

                # 显示矩形区域    
                newmask = cv2.rectangle(newmask, (left,up), (left + width, up + width), (0, 255, 0), 3)
                
                # 分割出矩形区域
                imgTest = lab[up:up+width, left:left+width]
                if imgTest.shape[0] > 0 and imgTest.shape[1] > 0:
                    # 识别字符
                    imgTest = cv2.resize(imgTest, (28, 28), interpolation=cv2.INTER_CUBIC)                
                    imgTest[imgTest < threshold] = 0
                    imgTest = imgTest / np.max(imgTest[:])                
                    imgTest = changeDim(imgTest)
                    resTest = np.argmax(model.predict(imgTest))
                    # 显示识别结果
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if resTest != 10:
                        newmask =  cv2.putText(newmask, str(resTest), (left, up), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        img_copy = img.copy()


# 初始画布
img = np.zeros((1024, 1024, 3), np.uint8) + 255
# 画布副本
img_b = img.copy()

# setting
flag = False
drawing = False  # 是否开始画图
start = (-1, -1)
threshold = 150 #字符分割阈值
model =load_model('model.h5')

cv2.namedWindow('Digital Recognition')
cv2.setMouseCallback('Digital Recognition', DigitalRecognition)

while(True):
    cv2.imshow('Digital Recognition', img)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
