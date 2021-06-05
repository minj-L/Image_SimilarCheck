import cv2, numpy as np
import matplotlib.pylab as plt

trackbar_name = 'fade' #트랙바

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    # ---③ addWeighted() 함수로 알파 블렌딩 적용
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow('cv2.addWeighted', dst)

alpha = 0.5 # 합성에 사용할 알파 값

img1 = cv2.imread('./Image_SimilarCheck/images/original.jpg')
img2 = cv2.imread('./Image_SimilarCheck/images/retouch.jpg')

# 블렌딩하는 두 이미지의 크기를 같게 함
width = img1.shape[1]
height = img1.shape[0]
img2 = cv2.resize(img2, (width, height))

imgs = [img1, img2]
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
cv2.imshow('cv2.addWeighted', img1)
cv2.createTrackbar(trackbar_name, 'cv2.addWeighted', 0, 100, onChange)
cv2.waitKey(0)
cv2.destroyAllWindows()