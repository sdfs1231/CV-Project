import cv2
import numpy as np

def get_features(Object):
    gray=cv2.cvtColor(Object,cv2.COLOR_RGB2GRAY)
    SIFT=cv2.xfeatures2d.SIFT_create()
    kps,features=SIFT.detectAndCompute(gray,None)
    kps = np.float32([kp.pt for kp in kps])
    return kps,features

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

# Object=cv2.imread('3.jpg')
# Canvas=cv2.imread('4.jpg')
# find_features_on_canvas(Object,Canvas,0.75,4)
# exit()

def draw_lines(Object,Canvas,ratio,threshold):
    heightA, widthA = Object.shape[:2]
    heightB, widthB = Canvas.shape[:2]
    kpsA,featuresA=get_features(Object)
    kpsB,featuresB=get_features(Canvas)

    vis=np.zeros((max(heightA,heightB),widthA+widthB,3),np.uint8)
    vis[0:heightA,0:widthA]=Object
    vis[0:heightB, widthA:]=Canvas
    result=find_features_on_canvas(Object,Canvas,ratio,threshold)
    matches=result[0]
    status=result[2]
    for ((trainIdx,queryIdx),s) in zip(matches,status):

        if s==1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0])+widthA, int(kpsB[trainIdx][1]))
            cv2.line(vis,ptA,ptB,(255,255,0),1)
    return vis

#find the perspective img
def perspective_transform(Object,Canvas,ratio,threshold):
    result=find_features_on_canvas(Object,Canvas,ratio,threshold)
    if result==None:
        return None
    (matches,H,status)=result
    # print(imgA.shape[0],imgA.shape[1],imgB.shape[0],imgB.shape[1])
    tranform_img=cv2.warpPerspective(Object,H,(Canvas.shape[1],Canvas.shape[0]))
    return tranform_img
#
# Object=cv2.imread('3.jpg')
# Canvas=cv2.imread('4.jpg')
# print(Object.shape)
# img_p=perspective_transform(Object,Canvas,0.75,4)
# cv2.imwrite('perspective.jpg',img_p)
# exit()

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


def find_the_box(Object,Canvas,ratio,threshold):
    Oh,Ow=Object.shape[:2]
    (matches,H,status)=find_features_on_canvas(Object,Canvas,ratio,threshold)
    box=[]
    box.append(find_cord(0,0,H))
    box.append(find_cord(Ow-1,0,H))
    box.append(find_cord(0,Oh-1,H))
    box.append(find_cord(Ow-1,Oh-1,H))
    print(box)
    cv2.line(Canvas,box[0],box[1],(0,255,0),1)
    cv2.line(Canvas,box[0],box[2],(0,255,0),1)
    cv2.line(Canvas,box[1],box[3],(0,255,0),1)
    cv2.line(Canvas,box[2],box[3],(0,255,0),1)
    return Canvas

Object=cv2.imread('3.jpg')
Canvas=cv2.imread('4.jpg')
ratio=0.75
threshold=4
box_img=find_the_box(Object,Canvas,ratio,threshold)
# vis=draw_lines(Object,Canvas,0.75,4)
# cv2.imwrite('lines.jpg',vis)
cv2.imwrite('box.jpg',box_img)
# cv2.imshow('test',box_img)
# key=cv2.waitKey()
# if key==27:
#     cv2.destroyAllWindows()
