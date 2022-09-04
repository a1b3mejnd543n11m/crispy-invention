from pickle import NONE
import cv2
import numpy as np
from collections import deque
bluelower=np.array([100,60,60])
blueupper=np.array([140,255,255])
kernel=np.ones((5,5),np.uint8)
bpoint=[deque(maxlen=512)]
gpoint=[deque(maxlen=512)]
rpoint=[deque(maxlen=512)]
ypoint=[deque(maxlen=512)]
bindex=0
gindex=0
rindex=0
yindex=0
colours=[(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colourindex=0
paintWindow=np.zeros((471,636,3))+255
paintWindow=cv2.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)
paintWindow=cv2.rectangle(paintWindow,(160,1),(255,65),colours[0],-1)
paintWindow=cv2.rectangle(paintWindow,(275,1),(370,65),colours[1],-1)
paintWindow=cv2.rectangle(paintWindow,(390,1),(480,65),colours[2],-1)
paintWindow=cv2.rectangle(paintWindow,(505,1),(600,65),colours[3],-1)

cv2.putText(paintWindow,"clear all",(49,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow,"blue",(185,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow,"green",(298,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow,"red",(420,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow,"yellow",(520,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)
cv2.namedWindow('paint',cv2.WINDOW_AUTOSIZE)
video=cv2.VideoCapture(0)


while True:
    hasframe,frame=video.read()
    frame=cv2.flip(frame,1)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame=cv2.rectangle(frame,(40,1),(140,65),(122,122,122),-1)
    frame=cv2.rectangle(frame,(160,1),(255,65),colours[0],-1)
    frame=cv2.rectangle(frame,(275,1),(370,65),colours[1],-1)
    frame=cv2.rectangle(frame,(390,1),(485,65),colours[2],-1)
    frame=cv2.rectangle(frame,(505,1),(600,65),colours[3],-1)
    cv2.putText(frame,"clear all",(49,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"blue",(185,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"green",(298,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"red",(420,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"yellow",(520,33),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)
    if not hasframe:
        break
    bluemask=cv2.inRange(hsv,bluelower,blueupper)
    bluemask=cv2.erode(bluemask,kernel,iterations=2)
    bluemask=cv2.morphologyEx(bluemask,cv2.MORPH_OPEN,kernel)
    bluemask=cv2.dilate(bluemask,kernel,iterations=1)
    (cnts,_)=cv2.findContours(bluemask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    centre=NONE
    if len(cnts)>0:
        cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        ((x,y),radius)=cv2.minEnclosingCircle(cnt)
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        M=cv2.moments(cnt)
        centre=(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        if centre[1]<=65:
            if 40<=centre[0]<=140:
                bpoint=[deque(maxlen=512)]
                gpoint=[deque(maxlen=512)]
                rpoint=[deque(maxlen=512)]
                ypoint=[deque(maxlen=512)]
                bindex=0
                rindex=0
                gindex=0
                yindex=0
                paintWindow[67:,:,:]=255
            elif 160<=centre[0]<=255:
                colourindex=0
            elif 275<=centre[0]<=370:
                colourindex=1     
            elif 390 <=centre[0]<=485:
                colourindex=2
            elif  505<=centre[0]<=600      :
                colourindex=3
        else :
            if colourindex==0:
                bpoint[bindex].appendleft(centre)    
            if colourindex==1:
                gpoint[gindex].appendleft(centre)
            if colourindex==2:
                rpoint[rindex].appendleft(centre) 
            if colourindex==3:
                ypoint[yindex].appendleft(centre)  
    else:
        bpoint.append(deque(maxlen=512))  
        bindex+=1 
        gpoint.append(deque(maxlen=512))  
        gindex+=1 
        rpoint.append(deque(maxlen=512))  
        rindex+=1 
        ypoint.append(deque(maxlen=512))  
        yindex+=1   
    points=[bpoint,gpoint,rpoint,ypoint]
    for i in range (len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if (points[i][j][k-1] is None or points[i][j][k]is None):
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colours[i],2)
                cv2.line(paintWindow,points[i][j][k-1],points[i][j][k],colours[i],2)
               

    cv2.imshow('frame',frame)
    cv2.imshow('paint',paintWindow)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break