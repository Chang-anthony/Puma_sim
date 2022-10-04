#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""
import numpy as np
import math

#自定義numpy 函式 
np.set_printoptions(precision=3,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg


class Puma():
    def __init__(self):
        self.dt = 0.04 #40ms
        self.src = np.eye(4)
        self.p1 = np.eye(4)
        self.p2 = np.eye(4)
        self.p3 = np.eye(4)
        self.p4 = np.eye(4)
        self.p5 = np.eye(4)
        self.end = np.eye(4)
        
        self.degs = np.zeros(6)
        self.bufferdegs = np.zeros(6)
        self.ResetPos()
    
    def Init(self):
        deg = np.zeros(6)
        src_pos = np.array([0,0,40])
        self.src[:3,3] = src_pos
        Pos = self.FK(deg)

        return Pos,deg

    def InitPos(self):
        self.src = np.eye(4)
        Pos,degs = self.Init()
        self.bufferdegs = degs

        self.FixedSrc(Pos)
        
        return degs

    def ResetPos(self):
        '''
            return degs Pos
        '''
        self.src = np.eye(4)
        src_pos = np.array([0,0,40])
        self.src[:3,3] = src_pos

        Pos,degs = self.workready()
        self.bufferdegs = degs
        self.FixedSrc(Pos)

        return degs,Pos

    def FixedSrc(self,Pos):
        self.p1 = self.src @ Pos[0]
        self.p2 = self.src @ Pos[1] 
        self.p3 = self.src @ Pos[2]
        self.p4 = self.src @ Pos[3] 
        self.p5 = self.src @ Pos[4] 
        self.end = self.src @ Pos[5]  

    def workready(self):
        deg = np.zeros(6)

        deg[0] = 0
        deg[1] = 10
        deg[2] = 20
        deg[3] = 30
        deg[4] = 10
        deg[5] = 0

        Pos = self.FK(deg)

        return Pos,deg

    def TransXYZ(self,dx = 0,dy = 0,dz = 0):
        trans = np.eye(4)

        trans[0,3] = dx
        trans[1,3] = dy
        trans[2,3] = dz

        return trans

    def RotX(self,deg):
        rad = degTrad(deg)
        rotx = np.eye(4)

        rotx[1,1] = cos(rad)
        rotx[1,2] = -sin(rad)
        rotx[2,1] = sin(rad)
        rotx[2,2] = cos(rad)

        return rotx

    def RotY(self,deg):
        rad = degTrad(deg)
        roty = np.eye(4)

        roty[0,0] = cos(rad)
        roty[0,2] = sin(rad)
        roty[2,0] = -sin(rad)
        roty[2,2] = cos(rad)

        return roty
    
    def RotZ(self,deg):
        rad = degTrad(deg)
        rotz = np.eye(4)

        rotz[0,0] = cos(rad)
        rotz[0,1] = -sin(rad)
        rotz[1,0] = sin(rad)
        rotz[1,1] = cos(rad)

        return rotz

    def RotXYZ(self,rx = 0,ry = 0,rz = 0):
        RotX = self.RotX(rx)
        RotY = self.RotY(ry)
        RotZ = self.RotZ(rz)

        return RotX @ RotY @ RotZ

    # rotx(alpha)Transx(a)Transz(d)rotz(θ)
    def RTTR(self,dh):
        rotx = self.RotX(dh[0])
        transx = self.TransXYZ(dh[1],0,0)
        transz = self.TransXYZ(0,0,dh[2])
        rotz = self.RotZ(dh[3])
        RTTR = np.eye(4)

        RTTR = (((rotx @ transx) @ transz) @ rotz)

        return RTTR

    def Get_Vector(self,crood):
        '''
            input crood
            output 1D 1 * 12 array

            ex:
            4X4 crood
            [Xx,Yx,Zx,Px]
            [Xy,Yy,Zy,Py]
            [Xz,Yz,Zz,Pz]
            [0,0,0,1]

            output = [Xx,Xy,Xz,Yx,Yy,Yz,Zx,Zz,Px,Py,Pz]
        '''
        return crood[:,:3].T.reshape(-1)

    def Get_J(self,Pos):
        J = np.zeros((12,6))
        end_p = Pos[-1][0:3,3]
        for i in range(6):
            rn = end_p - Pos[i][0:3,3]
            #因為puma 560 都是旋轉z軸
            zn = Pos[i][0:3,2]
            
            #wx
            J[0:3,i] = np.cross(zn,Pos[-1][0:3,0])
            #wy 
            J[3:6,i] = np.cross(zn,Pos[-1][0:3,1])
            #wz
            J[6:9,i] = np.cross(zn,Pos[-1][0:3,2])
            #V
            J[9:12,i] = np.cross(zn,rn)
       
        return J
        
    def FK(self,degs):
        '''
            for introdution book
            return Pos list
        '''
        
        T1 = self.RTTR([0,0,0,degs[0]])
        T2 = self.RTTR([-90,0,0,degs[1]])
        T3 = self.RTTR([0,20,0,degs[2]])
        T4 = self.RTTR([-90,0,20,degs[3]])
        T5 = self.RTTR([90,0,0,degs[4]])
        T6 = self.RTTR([-90,0,0,degs[5]])

        P1 = T1 
        P2 = P1 @ T2 
        P3 = P2 @ T3 
        P4 = P3 @ T4
        P5 = P4 @ T5
        end = P5 @ T6

        # T1 = self.RotZ(degs[0])
        # T2 = self.RotX(-90)
        # T3 = self.RotZ(degs[1])
        # T4 = self.TransXYZ(20,0,20)
        # T5 = self.RotZ(degs[2])
        # T6 = self.TransXYZ(10,0,0)
        # T7 = self.RotX(-90)
        # T8 = self.RotZ(degs[3])
        # T9 = self.RotX(90)
        # T10 = self.RotZ(degs[4])
        # T11 = self.RotX(-90)
        # T12 = self.RotZ(degs[5])



        # P1 = T1 
        # P2 = P1 @ T2 @ T3 
        # P3 = P2 @ T4 @ T5
        # P4 = P3 @ T6 @ T7 @ T8
        # P5 = P4 @ T9 @ T10
        # end = P5 @ T11 @ T12

        return [P1,P2,P3,P4,P5,end] 

    def IK(self,goal_end,src):
        '''
            goal = wTgoal
            Pos[-1] = srcTend
            src = wTsrc
        '''
        sTw = np.linalg.inv(src)
        sTgoal = sTw @ goal_end 


        goal_degs = np.zeros(6)
        goal_degs[0] = 0
        goal_degs[1] = 0
        goal_degs[2] = 0
        goal_degs[3] = 0
        goal_degs[4] = 10
        goal_degs[5] = 0

        iter = 1000
        aplha = 0.9

        while True:
            # V = JW
            # w = J(-1)*V
            iter -= 1
            Pos_end = self.FK(goal_degs)
            V = sTgoal - Pos_end[-1]
            V =  V[:,0:3].T.reshape(-1)
            error = np.sum(V ** 2)
            if error < 0.0001 or iter <= 0:
                break

            J = self.Get_J(Pos_end)
            wi = np.linalg.pinv(J) @ V
            goal_degs = goal_degs + aplha * radTdeg(wi)

        return goal_degs

    def Get_Matrix(self,A,B,lamda):
        '''
            return [A ~ B] matrix lamda
        '''

        D = np.linalg.inv(A) @ B
        theta = math.acos((D[0,0] + D[1,1] + D[2,2]-1)/2)

        if round(sin(theta * pi),4) != round(0.000,4):
            u = 2 * sin(theta)
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u
        else:
            u = 0.001
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u

        OUT = np.eye(4)

        dx = lamda * D[0,3]
        dy = lamda * D[1,3]
        dz = lamda * D[2,3]

        C = cos(lamda * theta)
        S = sin(lamda * theta)  
        V = 1.0 - C


        OUT[0,0] = kx * kx * V + C
        OUT[0,1] = kx * ky * V - kz * S
        OUT[0,2] = kx * kz * V + ky * S
        OUT[0,3] = dx
        
        OUT[1,0] = kx * ky * V + kz * S 
        OUT[1,1] = ky * ky * V + C
        OUT[1,2] = ky * kz * V - kx * S
        OUT[1,3] = dy

        OUT[2,0] = kx * kz * V - ky * S
        OUT[2,1] = ky * kz * V + kx * S
        OUT[2,2] = kz * kz * V + C
        OUT[2,3] = dz

        #Convert OUT TO PA frame
        OUT = A @ OUT 

        return OUT

    def Get_Matrix_Trajectory(self,A,B,step):
        '''
            return mats [A ~ B] lens step +1
        '''

        D = np.linalg.inv(A) @ B
        theta = math.acos((D[0,0] + D[1,1] + D[2,2]-1)/2)

        if round(sin(theta * pi),4) != round(0.000,4):
            u = 2 * sin(theta)
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u
        else:
            u = 0.001
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u

        OUTs= np.zeros((step + 1,4,4))
        rates = np.arange(0,1+1/step,1/step)
        us = np.ones((step+1))
        Cs = cos(rates)
        Ss = sin(rates)
        Vs = 1 - Cs

        dxs = rates * D[0,3]
        dys = rates * D[1,3]
        dzs = rates * D[2,3]

        OUTs[:,0,0] = kx * kx * Vs + Cs
        OUTs[:,0,1] = kx * ky * Vs - kz * Ss
        OUTs[:,0,2] = kx * kz * Vs + ky * Ss
        OUTs[:,0,3] = dxs 

        OUTs[:,1,0] = kx * ky * Vs + kz * Ss
        OUTs[:,1,1] = ky * ky * Vs + Cs
        OUTs[:,1,2] = ky * kz * Vs - kx * Ss
        OUTs[:,1,3] = dys

        OUTs[:,2,0] = kx * kz * Vs - ky * Ss 
        OUTs[:,2,1] = ky * kz * Vs - kx * Ss 
        OUTs[:,2,2] = kz * kz * Vs + Cs 
        OUTs[:,2,3] = dzs

        OUTs[:,3,3] = us
        #convert to A frame 
        OUTs = A @ OUTs 
        return OUTs


if __name__ == "__main__":
    test = Puma()
    test.InitPos()

    print("good")