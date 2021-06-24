import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    # ---③ addWeighted() 함수로 알파 블렌딩 적용
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow('cv2.addWeighted', dst)

alpha = 0.5 # 합성에 사용할 알파 값

img = cv2.imread('./Image_SimilarCheck/images/original_3.jpg')
img2 = cv2.imread('./Image_SimilarCheck/images/retouch_3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 블렌딩하는 두 이미지의 크기를 같게 함
width = img.shape[1]
height = img.shape[0]
img2 = cv2.resize(img2, (width, height))

img = img.reshape((img.shape[0] * img.shape[1], 3)) # height, width 통합

k = 1 # 예제는 5개로 나누겠습니다
clt = KMeans(n_clusters = k)
clt.fit(img)

for center in clt.cluster_centers_:
    print(center)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist
hist = centroid_histogram(clt)
print(hist)

if 233 in center:
    trackbar_name = 'fade' #트랙바

    imgs = [img, img2]
    hists = []
    for i, img in enumerate(imgs) :
        #---① 각 이미지를 HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #---② H,S 채널에 대한 히스토그램 계산
        hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
        #---③ 0~1로 정규화
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)

    query = hists[0]
    methods = {'CORREL' :cv2.HISTCMP_CORREL}
    for j, (name, flag) in enumerate(methods.items()):
        for i, (hist, img) in enumerate(zip(hists, imgs)):
            #---④ 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
            ret = cv2.compareHist(query, hist, flag)
            if flag == cv2.HISTCMP_INTERSECT: #교차 분석인 경우 
                ret = ret/np.sum(query)        #비교대상으로 나누어 1로 정규화
            print("accuracy:%.2f"% (ret), end='\t')
        print()
    cv2.imshow('cv2.addWeighted', img)
    cv2.createTrackbar(trackbar_name, 'cv2.addWeighted', 0, 100, onChange)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
def detect_color_image(img, MSE_cutoff=22):
    img = cv2.imread('./Image_SimilarCheck/images/original_3.jpg')
    bands = img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = img.resize((img.shape[1], img.shape[0]))
        SSE, bias = 0, [0, 0, 0]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (img.shape[1], img.shape[0])
        if MSE <= MSE_cutoff:
            print("No Color"),
        else:
            print("Color")
    if len(bands) == 1:
        print("Black and white")
"""