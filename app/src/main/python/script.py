import numpy as np
import math
import cv2
from imutils import paths


def Angle(x1,y1,x2,y2):
    dx=abs(x1-x2)
    dy=abs(y1-y2)
    if x1==x2:
        theta=90
    elif y1==y2:
        theta=0
    elif ((x1>x2)&(y1<y2))|((x1<x2)&(y1>y2)):
        theta=(180/math.pi)*math.atan(dy/dx) 
    else:
        theta=180-(180/math.pi)*math.atan(dy/dx)
    return theta


def Orientation_distance(x_M1,y_M1,x_M2,y_M2,x_T1,y_T1,x_T2,y_T2,W):
    W=30
    thetaM=Angle(x_M1,y_M1,x_M2,y_M2)   
    thetaT=Angle(x_T1,y_T1,x_T2,y_T2)   
    if ((thetaM<(90)) & (thetaT<(90)))|((thetaM>(90)) & (thetaT>(90))):
        d_theta=math.pow(abs(thetaM-thetaT),2)/W
    elif abs(thetaM-thetaT)<(90):
        d_theta=math.pow(abs(thetaM-thetaT),2)/W
    else:
        d_theta=(180-math.pow(abs(thetaM-thetaT),2))/W
    return [thetaM,thetaT,d_theta]     


def Parallel_distance(x_drvt,x_1p,y_1p,x_2p,y_2p,x_1n,y_1n,x_2n,y_2n):
    if x_drvt==0:
        if y_1p<y_2p: 
            left_long=[x_1p, y_1p]
            right_long=[x_2p, y_2p]
        else:
            right_long=[x_1p, y_1p]
            left_long=[x_2p, y_2p]
    elif x_1p<x_2p:
            left_long=[x_1p, y_1p]
            right_long=[x_2p, y_2p]
    else:
            right_long=[x_1p, y_1p]
            left_long=[x_2p, y_2p]
    
    if x_drvt==0:
        if y_1n<y_2n: 
            left_short=[x_1n, y_1n]
            right_short=[x_2n, y_2n]
        else:
            right_short=[x_1n, y_1n]
            left_short=[x_2n, y_2n]
    elif x_1n<x_2n:
            left_short=[x_1n, y_1n]
            right_short=[x_2n, y_2n] 
    else:
            right_short=[x_1n, y_1n]
            left_short=[x_2n, y_2n] 
    
    d_parallel_left=math.sqrt(math.pow(left_long[0] - left_short[0],2) + math.pow(left_long[1]-left_short[1],2))
    d_parallel_right=math.sqrt(math.pow(right_long[0] - right_short[0],2) + math.pow(right_long[1]-right_short[1],2))
    
    if x_drvt==0:
        if (left_short[1]>=left_long[1]) & (right_short[1]<=right_long[1]):
            Paradis=0 
        else:
            Paradis=min(d_parallel_left, d_parallel_right)
    elif (left_short[0]>=left_long[0]) & (right_short[0]<=right_long[0]):
            Paradis=0
    else:
            Paradis=min(d_parallel_left,d_parallel_right)
    return Paradis


def Projecting_line(x_drvt,y_drvt,x_nmvt,y_nmvt,x_midpointshorterline,y_midpointshorterline,x1_old,y1_old,x2_old,y2_old):
    if y_drvt==0:
        y_1p=y_midpointshorterline
        y_2p=y_midpointshorterline
        x_1p=x1_old
        x_2p=x2_old
    else:
        if x_drvt==0:
            y_1p=y1_old
            y_2p=y2_old
            x_1p=x_midpointshorterline
            x_2p=x_midpointshorterline
        else:
            y_1p=(x_midpointshorterline-x1_old-((x_drvt/y_drvt)*y_midpointshorterline)+((x_nmvt/y_nmvt)*y1_old))/((x_nmvt/y_nmvt)-(x_drvt/y_drvt))
            x_1p=x_midpointshorterline+((x_drvt/y_drvt)*(y_1p-y_midpointshorterline))
            y_2p=(x_midpointshorterline-x2_old-((x_drvt/y_drvt)*y_midpointshorterline)+((x_nmvt/y_nmvt)*y2_old))/((x_nmvt/y_nmvt)-(x_drvt/y_drvt))
            x_2p=x_midpointshorterline+((x_drvt/y_drvt)*(y_2p-y_midpointshorterline))
    return [x_1p,y_1p,x_2p,y_2p]


def Rotating_shorter_line(x_drvt,y_drvt,x_midpointshorterline,y_midpointshorterline,haftlengthshoterline):
    if y_drvt==0:
        x_1n=x_midpointshorterline+haftlengthshoterline
        x_2n=x_midpointshorterline-haftlengthshoterline
        y_1n=y_midpointshorterline
        y_2n=y_midpointshorterline
    elif x_drvt==0:
            x_1n=x_midpointshorterline
            x_2n=x_midpointshorterline
            y_1n=y_midpointshorterline+haftlengthshoterline
            y_2n=y_midpointshorterline-haftlengthshoterline                      
    else:
            A=math.sqrt((math.pow(haftlengthshoterline,2)*math.pow(x_drvt,2))/(math.pow(x_drvt,2) + math.pow(y_drvt,2)))
            x_1n=x_midpointshorterline+A
            x_2n=x_midpointshorterline-A
            y_1n=y_midpointshorterline+((y_drvt/x_drvt)*A)
            y_2n=y_midpointshorterline+((y_drvt/x_drvt)*(-A))
    return [x_1n,y_1n,x_2n,y_2n]   


def Dist(line1,line2,Na,Np,W):

    M1=[line1[0], line1[1]]
    M2=[line1[2], line1[3]]
    lM = math.sqrt(math.pow(M1[0]-M2[0],2) + math.pow(M1[1]-M2[1],2))  
    T1=[line2[0], line2[1]]
    T2=[line2[2], line2[3]]

    #######  Calculate the value of Orientation distance #######
        #######  Parallel distance and Perpendicular distance  #######
        
        ##### Orientation distance #####   
    [thetaM, thetaT, d_theta] = Orientation_distance(M1[0],M1[1],M2[0],M2[1],T1[0],T1[1],T2[0],T2[1],W)
    ##### Find the midpoint of two lines #####
    Mmp=[(M1[0]+ M2[0])/2, (M1[1]+ M2[1])/2]
    Tmp=[(T1[0]+ T2[0])/2, (T1[1]+ T2[1])/2]
    ##### Half length of each line #####
    lM_mp=math.sqrt(math.pow(Mmp[0]- M1[0],2) + math.pow(Mmp[1]-M1[1],2))
    lT_mp=math.sqrt(math.pow(Tmp[0]- T1[0],2) + math.pow(Tmp[1]-T1[1],2))
    ##### Rotate the shorter line #####
    if (d_theta==0)&(Mmp[0]==Tmp[0])&(Mmp[1]==Tmp[1])&(lM_mp==lT_mp):
        d_perpendicular=0
        d_parallel=0
    elif lM_mp>=lT_mp:
        drvtM = [M1[0]-M2[0], M1[1]-M2[1]]   # Direction vector of line #
        nmvtM = [-drvtM[1], drvtM[0]]  # Normal vector of line #
        a = nmvtM[0]
        b = nmvtM[1]
        a1 = drvtM[0]
        b1 = drvtM[1]
        # Find the new coordinates of shorter line #
        [x_T1n,y_T1n,x_T2n,y_T2n] = Rotating_shorter_line(a1,b1,Tmp[0],Tmp[1],lT_mp)                          
        T1n=[x_T1n, y_T1n]
        T2n=[x_T2n, y_T2n]
        # M line is projected on T line #
        [x_M1Tp,y_M1Tp,x_M2Tp,y_M2Tp] = Projecting_line(a1,b1,a,b,Tmp[0],Tmp[1],M1[0],M1[1],M2[0],M2[1])                                  
        M1Tp=[x_M1Tp, y_M1Tp]
        M2Tp=[x_M2Tp, y_M2Tp]
        ##### Perpendicular distance #####
        d_perpendicular=math.sqrt(math.pow(M1Tp[0]-M1[0],2) + math.pow(M1Tp[1]-M1[1],2))
        ##### Parallel distance ##### 
        d_parallel=Parallel_distance(a1,M1Tp[0],M1Tp[1],M2Tp[0],M2Tp[1],T1n[0],T1n[1],T2n[0],T2n[1])           
    else:
        drvtT=[T1[0]-T2[0], T1[1]-T2[1]]   # Direction vector of line #
        nmvtT=[-drvtT[1], drvtT[0]]  # Normal vector of line #
        a=nmvtT[0]
        b=nmvtT[1]
        a1=drvtT[0]
        b1=drvtT[1]
        ##### Find the new coordinates of shorter line #####
        [x_M1n,y_M1n,x_M2n,y_M2n] = Rotating_shorter_line(a1,b1,Mmp[0],Mmp[1],lM_mp)                                      
        M1n=[x_M1n, y_M1n]
        M2n=[x_M2n, y_M2n]
        ##### T line is projected on M line #####
        [x_T1Mp,y_T1Mp,x_T2Mp,y_T2Mp] = Projecting_line(a1,b1,a,b,Mmp[0],Mmp[1],T1[0],T1[1],T2[0],T2[1])                        
        T1Mp=[x_T1Mp, y_T1Mp]
        T2Mp=[x_T2Mp, y_T2Mp]
        ##### Perpendicular distance #####
        d_perpendicular=math.sqrt(math.pow(T1Mp[0]-T1[0],2) + math.pow(T1Mp[1]-T1[1],2))               
        ##### Parallel distance #####
        d_parallel=Parallel_distance(a1,T1Mp[0],T1Mp[1],T2Mp[0],T2Mp[1],M1n[0],M1n[1],M2n[0],M2n[1])

    Disline = math.sqrt(math.pow(d_theta,2) + math.pow(d_perpendicular,2) + math.pow(d_parallel,2))

    d_position_neighbor = math.sqrt(math.pow(Mmp[0]-Tmp[0],2) + math.pow(Mmp[1]-Tmp[1],2))
    d_angle_neighbor = abs(thetaM-thetaT)    
    if (d_position_neighbor<=Np)&(d_angle_neighbor<=Na):
        confi = 1
    else:
        confi = 0
    
    return [Disline,lM,confi]


def LHD(model,test,Wn,Kg,Kvt,k,W):
    V = 325
    lMS = 0
    lTS = 0
    SumM = 0
    SumT = 0
    #k = 1
    lineT = test
    lineM = model
    nM = 0
    nT = 0
    # T=dlmread('D:\Research\Matcode\RHD LHD\Bern\01\Adam1.ssfpM')
    # M=dlmread('D:\Research\Matcode\RHD LHD\Bern\02\Adam2.ssfpM')
    h_MT = np.zeros((len(lineM),3))
    h_TM = np.zeros((len(lineT),3))

    for i in range(len(lineM)):
        cmin = float('Inf')
        for j in range(len(lineT)):
            hM,lM,confiM = Dist(lineM[i,:],lineT[j,:],Kg,Kvt,W)
            if hM < cmin:
                cmin = hM
                b = confiM
        h_MT[i,0] = lM*cmin
        h_MT[i,1] = lM
        h_MT[i,2] = b

    hausdorff_MT=sum(h_MT[:,0])/sum(h_MT[:,1])
    # hausdorff_MT=SumM/lMS

    for i in range(len(lineT)):
        cmin = float('Inf')   
        for j in range(len(lineM)):
            hT,lT,confiT = Dist(lineT[i,:],lineM[j,:],Kg,Kvt,W)
            if hT < cmin:
                cmin = hT
                d = confiT

        h_TM[i,0] = lT*cmin
        h_TM[i,1] = lT
        h_TM[i,2] = d

    hausdorff_TM=sum(h_TM[:,0])/sum(h_TM[:,1])
    #hausdorff_TM=SumT/lTS;
    ##### Primary line segment Hausdorff distance (pLHD) #####
    LHD_primary_MT = max(hausdorff_TM, hausdorff_MT)

    ##### Number of disparity #####
    nM = sum(h_MT[:,2])
    nT = sum(h_TM[:,2])
    D_n = 1 - 0.5*(nM/len(lineM)+nT/len(lineT))

    ##### Hausdorff distance #####
    LHD_value=math.sqrt(math.pow(LHD_primary_MT,2) + math.pow(Wn*D_n,2))

    return LHD_value


def CreaMatrixFromssfpM(I):
    I_saved = np.ndarray(4, dtype=object)
    for i in range(len(I)):
        if (I(i,3) > 0)&(I(i+1,3) > 0):
            I_saved = np.array(I_saved, [I(i,2), I(i,1), I(i+1,2), I(i+1,1)])   

    return I_saved


def divto4sub(lineA2):
    B0 = []
    B1 = []
    B2 = []
    B3 = []
    for i in range(len(lineA2)):
        if (lineA2[i][0] >= 80)&(lineA2[i][1] >= 80)&(lineA2[i][2] >= 80)&(lineA2[i][3] >= 80):
            B0.append([lineA2[i][0], lineA2[i][1], lineA2[i][2], lineA2[i][3]])
        elif (lineA2[i][0] <= 80)&(lineA2[i][1] <= 80)&(lineA2[i][2] <= 80)&(lineA2[i][3] <= 80):
            B1.append([lineA2[i][0], lineA2[i][1], lineA2[i][2], lineA2[i][3]])
        elif (lineA2[i][0] <= 80)&(lineA2[i][1] >= 80)&(lineA2[i][2] <= 80)&(lineA2[i][3] >= 80):
            B2.append([lineA2[i][0], lineA2[i][1], lineA2[i][2], lineA2[i][3]])
        elif (lineA2[i][0] >= 80)&(lineA2[i][1] <= 80)&(lineA2[i][2] >= 80)&(lineA2[i][3] <= 80):
            B3.append([lineA2[i][0], lineA2[i][1], lineA2[i][2], lineA2[i][3]])
    B0 = line(B0)
    B1 = line(B1)
    B2 = line(B2)
    B3 = line(B3)
    return B0,B1,B2,B3


def giaodiem(points):
    tructung = []
    new_points = []
    for i in range(len(points)):
        x1_M = points[i][0][0]
        y1_M = points[i][0][1]
        x2_M = points[i][0][2]
        y2_M = points[i][0][3]
        if ((y1_M<80) & (80<y2_M))|((y2_M<80) & (80<y1_M)):
            y = 80
            if x1_M == x2_M :
                x = x1_M
            else :
                a = (y1_M - y2_M)/(x1_M - x2_M)
                b = y1_M - x1_M*(y1_M - y2_M)/(x1_M - x2_M)
                x = (y-b)/a
            tructung.append([x1_M, y1_M, x, y])
            tructung.append([x2_M, y2_M, x, y])
        else:
            tructung.append([x1_M, y1_M, x2_M, y2_M])
    for j in range(len(tructung)):
        x1_M = tructung[j][0]
        y1_M = tructung[j][1]
        x2_M = tructung[j][2]
        y2_M = tructung[j][3]
        if ((x1_M<80) & (80<x2_M))|((x2_M<80) & (80<x1_M)):
            x = 80
            a = (y1_M - y2_M)/(x1_M - x2_M)
            b = y1_M - x1_M*(y1_M - y2_M)/(x1_M - x2_M)
            y = a*x + b
            new_points.append([x1_M, y1_M, x, y])
            new_points.append([x2_M, y2_M, x, y])
        else:
            new_points.append([x1_M, y1_M, x2_M, y2_M])
    return new_points


def line(A):
    C = []
    for i in range(len(A)):
        if (A[i][0]==A[i][2])&(A[i][1]==A[i][3]):
            C.append([A[i][0], A[i][1], A[i][2], A[i][3]+1])
        else:
            C.append([A[i][0], A[i][1], A[i][2], A[i][3]])
    B = np.zeros((len(C),4), dtype=np.double)
    for i in range(len(C)):
        B[i,0] = C[i][0]
        B[i,1] = C[i][1]
        B[i,2] = C[i][2]
        B[i,3] = C[i][3]
    return B


def main():
    dataset = "D:\\HocApp\\DCLV\\data"
    test = "D:\\HocApp\\DCLV\\test"
    list_images = paths.list_images(dataset)
    list_images = list(list_images)
    list_imagesT = paths.list_images(test)
    list_imagesT = list(list_imagesT)

    # data = []
    labels = []
    for i in range(len(list_images)):
        # img = cv2.imread(list_images[i])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (160,160))
        # data.append(img)
        label = list_images[i].split("\\")[-2].split(".")[0]
        labels.append(label)
      
    Wn = 12 
    Kg = 5 
    Kvt = 30
    W = 25
    w1 = 0.4
    w2 = 0.1
    w3 = 0.4
    w4 = 0.1
    # for count1 in range(1):
    I_T = cv2.imread(list_imagesT[1])

    I_T = cv2.GaussianBlur(I_T, ksize=(5, 5), sigmaX=1, sigmaY=1)

    gray_T = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color -> gray
    edges_T = cv2.Canny(gray_T, 20, 60)
    img_thin_T = cv2.ximgproc.thinning(edges_T)  
    lines_T = cv2.HoughLinesP(img_thin_T, 180, np.math.pi/180, 50, None,0,0)
    f1 = giaodiem(lines_T)
    I_T0, I_T1, I_T2, I_T3 = divto4sub(f1)
    D = []
    for count2 in range(len(list_images)):
        I_M = cv2.imread(list_images[count2])
        I_M = cv2.GaussianBlur(I_M, ksize=(5, 5), sigmaX=1, sigmaY=1)
        gray_M = cv2.cvtColor(I_M, cv2.COLOR_BGR2GRAY) # color -> gray
        edges_M = cv2.Canny(gray_M, 20, 60)
        img_thin_M = cv2.ximgproc.thinning(edges_M)  
        lines_M = cv2.HoughLinesP(img_thin_M, 180, np.math.pi/180, 50, None,0,0)
        f2 = giaodiem(lines_M)
        I_M0, I_M1, I_M2, I_M3 = divto4sub(f2)

        LHD_value1 = LHD(I_T0,I_M0,Wn,Kg,Kvt,1,W)
        LHD_value2 = LHD(I_T1,I_M1,Wn,Kg,Kvt,1,W)
        LHD_value3 = LHD(I_T2,I_M2,Wn,Kg,Kvt,1,W)
        LHD_value4 = LHD(I_T3,I_M3,Wn,Kg,Kvt,1,W)        
        
        D1 = math.exp(-LHD_value1)
        D2 = math.exp(-LHD_value2)
        D3 = math.exp(-LHD_value3)
        D4 = math.exp(-LHD_value4)

        D.append(0.25*(w1*D1 + w2*D2 + w3*D3 + w4*D4))
    max_D = D[0]
    for m in range(1,len(D)):
        if max_D <= D[m]:
                max_D = D[m]
                countd = m
    label_predict = data_train[countd].split("\\")[-2].split(".")[0]
    return str(label_predict)
