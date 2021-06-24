import cv2
import numpy as np
import matplotlib.pyplot as plt

trackbar_name = 'fade' #트랙바

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    # ---③ addWeighted() 함수로 알파 블렌딩 적용
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow('cv2.addWeighted', dst)

alpha = 0.5 # 합성에 사용할 알파 값

img1 = cv2.imread('./Image_SimilarCheck/images/original_4.jpg')
img2 = cv2.imread('./Image_SimilarCheck/images/retouch_3.jpg')

img3 = cv2.imread('./Image_SimilarCheck/images/original_3.jpg')
img4 = cv2.imread('./Image_SimilarCheck/images/retouch_3.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 블렌딩하는 두 이미지의 크기를 같게 함
width = img1.shape[1]
height = img1.shape[0]
img2 = cv2.resize(img2, (width, height))

if img1.shape[2]==3 and img2.shape[2]==3: #img1,2가 모두 컬러이미지면(채널이 3이면) 피쳐매칭 실행하기
    detector = cv2.ORB_create() #특징점을 찾는 함수 ORB는 특징점을 찾는 알고리즘
    kp1, desc1 = detector.detectAndCompute(gray1, None) #detectorandcompute : 특징점검출과 특징 디스크립터 계산을 한번에 수행한다.
    kp2, desc2 = detector.detectAndCompute(gray2, None) #gray1,2는 입력이미지, None이므로 특징점 검출을 수행함
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #핵심 BFMatcher()을 통해 두 사진간 유사도가 가장 높은 키포인트 쌍들을 찾은 결과를
        #Norm_Hamming은 ORB와 같은 2진 문자열 기반 알고리즘에서 사용되어져야 한다. crosstype이 true인 이유는 보다 정확한 동일 특징점을 추출하기 위함이다.
    matches = matcher.match(desc1, desc2) #이 matches에 저장한다.

    matches = sorted(matches, key=lambda x:x.distance)

    #매칭 영역 원근 변환
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]) #원본좌표배열(src_pts) 좋은 매칭점의 queryIndex로 원본 사진의 좌표 구하기
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]) #결과좌표배열(dst_pts) 좋은 매칭점의 trainIndex로 대상 사진의 좌표 구하기

    #mtrx : 결과 변환 행렬, mask: 정상치 판별 결과, Nx1 배열 (0: 비정상치, 1: 정상치)
    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #findhomograhpy는 여러개의 점으로 근사 계산한 원근 변환 행렬을 반환
    #RANSAC알고리즘은 임의의 좌표만 선정해서 만족도를 구하는 방식인데, 이렇게 구한 만족도가 큰 것만 선정하여 근사 계산
    #정상치와 이상치를 구분해주는 mask를 반환하므로 올바른 매칭점과 나쁜 매칭점을 한번 더 구분할 수 있다.

    # 모든 매칭점과 정상치 비율
    accuracy=float(mask.sum()) / mask.size #mask.sum()은 정상치 판별결과list의 합계만 의미
    print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

    cv2.imshow('cv2.addWeighted', img1)
    cv2.createTrackbar(trackbar_name, 'cv2.addWeighted', 0, 100, onChange)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgs = [img3, img4]
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
cv2.imshow('cv2.addWeighted', img3)
cv2.createTrackbar(trackbar_name, 'cv2.addWeighted', 0, 100, onChange)
cv2.waitKey(0)
cv2.destroyAllWindows()