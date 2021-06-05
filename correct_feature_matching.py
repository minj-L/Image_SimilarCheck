import cv2, numpy as np

img1 = cv2.imread('./Image_SimilarCheck/images/original_4.jpg')
img2 = cv2.imread('./Image_SimilarCheck/images/retouch_4.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #본래 색상공간에서 다른 색상 공간으로 변환
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #img 1,2들을 그레이 색상으로 바꾸라는 의미

detector = cv2.ORB_create() #특징점을 찾는 함수 ORB는 특징점을 찾는 알고리즘
kp1, desc1 = detector.detectAndCompute(gray1, None) #detectorandcompute : 특징점검출과 특징 디스크립터 계산을 한번에 수행한다.
kp2, desc2 = detector.detectAndCompute(gray2, None) #gray1,2는 입력이미지, None이므로 특징점 검출을 수행함
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #핵심 BFMatcher()을 통해 두 사진간 유사도가 가장 높은 키포인트 쌍들을 찾은 결과를
#Norm_Hamming은 ORB와 같은 2진 문자열 기반 알고리즘에서 사용되어져야 한다. crosstype이 true인 이유는 보다 정확한 동일 특징점을 추출하기 위함이다.
matches = matcher.match(desc1, desc2) #이 matches에 저장한다.


matches = sorted(matches, key=lambda x:x.distance) #sorted는 모든 형식을 받아들여 정렬해주는 리스트
#추출된 특징점을 저장한 matches를 lambda키 값으로 정렬한다.
# 모든 매칭점 그리기 , 매칭 결과를 시각정으로 표현하는 함수
res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) #한쪽만 있는 매칭 결과 그리기 제외

#매칭 영역 원근 변환
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]) #원본좌표배열(src_pts) 좋은 매칭점의 queryIndex로 원본 사진의 좌표 구하기
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]) #결과좌표배열(dst_pts) 좋은 매칭점의 trainIndex로 대상 사진의 좌표 구하기

#mtrx : 결과 변환 행렬, mask: 정상치 판별 결과, Nx1 배열 (0: 비정상치, 1: 정상치)
mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #findhomograhpy는 여러개의 점으로 근사 계산한 원근 변환 행렬을 반환
#RANSAC알고리즘은 임의의 좌표만 선정해서 만족도를 구하는 방식인데, 이렇게 구한 만족도가 큰 것만 선정하여 근사 계산
#정상치와 이상치를 구분해주는 mask를 반환하므로 올바른 매칭점과 나쁜 매칭점을 한번 더 구분할 수 있다.

h,w = img1.shape[:2]
pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
dst = cv2.perspectiveTransform(pts,mtrx)
img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# 정상치 매칭만 그리기
matchesMask = mask.ravel().tolist() #정상치 판별 결과인 mask를 ravel로 1차원 배열로 만들어 준 뒤 tolist를 목록으로 변환해 준다.
res2 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                    matchesMask = matchesMask, #정상치 매칭만을 출력
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 모든 매칭점과 정상치 비율
accuracy=float(mask.sum()) / mask.size #mask.sum()은 정상치 판별결과list의 합계만 의미
print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
                 
cv2.imshow('Matching-All', res1)
cv2.imshow('Matching-Inlier ', res2)
cv2.waitKey()
cv2.destroyAllWindows()