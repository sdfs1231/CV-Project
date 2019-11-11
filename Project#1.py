import cv2
import numpy as np
import random

def get_features(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    sift=cv2.xfeatures2d.SIFT_create()
    kps,des=sift.detectAndCompute(gray,None)

    #convert to numpy array
    kps=np.float32([kp.pt for kp in kps])
    return kps,des

# img=cv2.imread('lenna.jpg')
# kps,des=get_features(img)


def match_keypoints(imgA,imgB,ratio,threshold):
    #get the features
    kpsA,featuresA=get_features(imgA)
    kpsB,featuresB=get_features(imgB)


    #create matcher
    matcher=cv2.DescriptorMatcher_create('BruteForce')

    #initialize the matches points
    unprocessedMatches=matcher.knnMatch(featuresA,featuresB,2)

    matches=[]

    for m in unprocessedMatches:
        #if the distance
        if len(m)==2 and m[0].distance<m[1].distance*ratio:
            matches.append((m[0].trainIdx,m[0].queryIdx))

    #need at leat 4pair points to do perspective change
    if len(matches)>4:
        ptsA=np.float32([kpsA[i] for (_,i) in matches])
        ptsB=np.float32([kpsB[i] for (i,_) in matches])

        #compute the homography
        H,status=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,threshold)

        return (matches,H,status)
    else:
        return None

def draw_lines(imgA,imgB,kpsA,kpsB,ratio,threshold):
    heightA, widthA = imgA.shape[:2]
    heightB, widthB = imgB.shape[:2]
    #the sticher visible img,notice the imgB is on the right side
    vis=np.zeros((max(heightA,heightB),widthA+widthB,3),np.uint8)
    vis[0:heightB,0:widthB]=imgB
    vis[0:heightB, widthB:]=imgA
    result=match_keypoints(imgA,imgB,ratio,threshold)
    matches=result[0]
    status=result[2]

    #look up the whole imgA and imgB
    for ((trainIdx,queryIdx),s) in zip(matches,status):
        #s==1 means the tow points matches
        if s==1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0])+widthA, int(kpsB[trainIdx][1]))
            cv2.line(vis,ptA,ptB,(255,255,0),1)
    return vis

def perspective_transform(imgA,imgB,ratio,threshold):
    result=match_keypoints(imgA,imgB,ratio,threshold)
    if result==None:
        return None
    (matches,H,status)=result
    # print(imgA.shape[0],imgA.shape[1],imgB.shape[0],imgB.shape[1])
    tranform_img=cv2.warpPerspective(imgA,H,(imgA.shape[1]+imgA.shape[1],imgB.shape[0]))
    return tranform_img

#pack up all funtions
def sticher(imgA,imgB,ratio,threshold):
    result=perspective_transform(imgA,imgB,ratio,threshold)
    result[0:imgB.shape[0],0:imgB.shape[1]]=imgB
    return result

imgA=cv2.imread('2.jpg')
imgB=cv2.imread('1.jpg')
result=sticher(imgA,imgB,0.95,4)
cv2.imshow('result',result)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
cv2.imwrite('result.jpg',result)
# print(imgA.shape)

# img_p=perspective_transform(imgA,imgB,0.75,4)
#
# print(img_p.shape)
# cv2.imshow('transform_img',img_p)
# key=cv2.waitKey()
# if key==27:
#     cv2.destroyAllWindows()

# kpsA,_=get_features(imgA)
# kpsB,_=get_features(imgB)
#
# rawmatches=match_keypoints(imgA,imgB,0.9,4)
# print(rawmatches)
# vis=draw_lines(imgA,imgB,kpsA,kpsB,0.9,4)
# cv2.imshow('lines',vis)
# key=cv2.waitKey()
# if key==27:
#     cv2.destroyAllWindows()


