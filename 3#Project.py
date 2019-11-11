import cv2
import numpy as np

def get_features(Object):
    gray=cv2.cvtColor(Object,cv2.COLOR_RGB2GRAY)
    SIFT=cv2.xfeatures2d.SIFT_create()
    kps,features=SIFT.detectAndCompute(gray,None)
    kps = np.float32([kp.pt for kp in kps])
    return kps,features

def find_cord(x,y,M):
    divider=M[2][0]*x+M[2][1]*y+M[2][2]
    x1=(M[0][0]*x+M[0][1]*y+M[0][2])/divider
    y1=(M[1][0]*x+M[1][1]*y+M[1][2])/divider
    if x1>1080:
        x1=1080
    if y1>1440:
        y1=1440
    x1=int(x1)
    y1=int(y1)
    return x1,y1


def find_features_on_canvas(Object,Canvas,ratio,threshold):
    matches=[]
    kpsA,featuresA=get_features(Object)
    kpsB,featuresB=get_features(Canvas)
    matcher=cv2.DescriptorMatcher_create('BruteForce')
    rawmatches=matcher.knnMatch(featuresB,featuresA,2)#query,train

    for m in rawmatches:
        if len(m)==2 and m[0].distance<m[1].distance*ratio:
            matches.append((m[0].queryIdx,m[0].trainIdx))
    # print(matches)
    if len(matches)>4:
        ptsA=np.float32([kpsA[i] for (_,i) in matches])
        ptsB=np.float32([kpsB[i] for (i,_) in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, threshold)
        return (matches,H,status)
    return None

def find_the_box(Object,Canvas,ratio,threshold):
    Oh,Ow=Object.shape[:2]
    (matches,H,status)=find_features_on_canvas(Object,Canvas,ratio,threshold)
    box=[]
    box.append(find_cord(0,0,H))
    box.append(find_cord(Ow-1,0,H))
    box.append(find_cord(0,Oh-1,H))
    box.append(find_cord(Ow-1,Oh-1,H))
    cv2.line(Canvas,box[0],box[1],(0,255,0),1)
    cv2.line(Canvas,box[0],box[2],(0,255,0),1)
    cv2.line(Canvas,box[1],box[3],(0,255,0),1)
    cv2.line(Canvas,box[2],box[3],(0,255,0),1)
    return Canvas

Object=cv2.imread('3.jpg')
cama=cv2.VideoCapture('example.mp4')
while 1:
    s,frame=cama.read()
    box_frame=find_the_box(Object,frame,0.75,4)
    cv2.imshow('capture',box_frame)
    if cv2.waitKey(10) and 0xFF==ord('q'):
        break
cama.release()
cv2.destroyAllWindows()