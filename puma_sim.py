#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""

import imp
import math
import re
import sys 
import os 
import csv 
import os.path
from OpenGL.GL import * 
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5 import QtCore,QtGui,QtWidgets,QtOpenGL
from PyQt5.QtWidgets import QDialog,QApplication,QMessageBox,QMainWindow,QAction

import debugpy
import cv2
import numpy as np
import time
import queue
from Puma import Puma

#自定義numpy 函式 
np.set_printoptions(precision=3,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg

class OpenGL_widget(QThread):
    def __init__(self,robotpoints,puma:Puma):
        super().__init__()
        self.query_gl = robotpoints
        self.shadow_puma = puma     
        self.dx = 1
        self.x1 = 0
        self.ts = []

        self.src = np.eye(4)
        self.p1 = np.eye(4)
        self.p2 = np.eye(4)
        self.p3 = np.eye(4)
        self.p4 = np.eye(4)
        self.p5 = np.eye(4)
        self.end = np.eye(4)

        self.stop = True
        self.window = None

        '''
            cammat 攝影機其次座標矩陣
            campos 攝影機在世界座標的x,y,z位置
            camlook 攝影機看向的世界座標點
        '''
        self.cammat = np.eye(4)
        self.cammat[:3,3] = np.array([100,100,100])
        self.campos = self.cammat[0:3,3]
        self.camlook = np.array([0,0,60])

        P = self.cammat[0:3,3] 
        A = self.camlook
        Zaxis = np.array([0,0,1])

        forward = (A-P) 
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward,Zaxis)
        right = right / np.linalg.norm(right)
        Top = np.cross(right,forward)
        Top = Top / np.linalg.norm(Top)


        '''
            camera Xaxis right
            camera Yaxis -Top
            camera Zaxis forward
        '''

        self.cammat[:3,0] = right
        self.cammat[:3,1] = -Top
        self.cammat[:3,2] = forward

    def close(self):
        if self.window:
            print("exit OpenGL")

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
        glutInitWindowPosition(0,0)
        glutInitWindowSize(640,360)

        self.window = glutCreateWindow("puma_sim")

        glutMouseWheelFunc(self.mouseWheel)
        glutKeyboardFunc(self.keyboard_envent)
        glutMouseFunc(self.mouse_event)
        glutDialsFunc(self.paintGL)
        glutIdleFunc(self.paintGL)

        glClearColor(0,0,0,1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glutMainLoop()
        glutMainLoopEvent()

    def TransXYZ(self,dx = 0, dy = 0,dz =0):
        Trans = np.eye(4)

        Trans[0,3] = dx
        Trans[1,3] = dy
        Trans[2,3] = dz

        return Trans

    def RotX(self,deg):
        rad = degTrad(deg)
        RotX = np.eye(4)

        RotX[1,1] = cos(rad)
        RotX[1,2] = -sin(rad)
        RotX[2,1] = sin(rad)
        RotX[2,2] = cos(rad)

        return RotX

    #右手座標係
    def RotY(self,deg):
        rad = degTrad(deg)
        RotY= np.eye(4)

        RotY[0,0] = cos(rad)
        RotY[0,2] = sin(rad)
        RotY[2,0] = -sin(rad)
        RotY[2,2] = cos(rad)
        
        return RotY

    def RotZ(self,deg):
        rad = degTrad(deg)
        RotZ = np.eye(4)
        
        RotZ[0,0] = cos(rad)
        RotZ[0,1] = -sin(rad)
        RotZ[1,0] = cos(rad)
        RotZ[1,1] = sin(rad)

        return RotZ

    def RotXYZ(self,rx = 0 ,ry = 0,rz = 0):
        RotX = self.RotX(rx)
        RotY = self.RotY(ry) 
        RotZ = self.RotZ(rz)

        return RotX @ RotY @ RotZ

    def paintGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45,640/360,0.01,500)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.campos = self.cammat[:3,3]
        self.camlook = np.array(self.src[:3,3])

        P = self.cammat[:3,3]
        A = self.camlook
        forward = (A-P)
        Zaxis = np.array([0,0,1])

        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward,Zaxis)
        right = right / np.linalg.norm(right)
        Top = np.cross(right,forward)
        Top = Top / np.linalg.norm(Top)

        self.cammat[:3,0] = right
        self.cammat[:3,1] = -Top
        self.cammat[:3,2] = forward

        gluLookAt(P[0],P[1],P[2],A[0],A[1],A[2],Top[0],Top[1],Top[2])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        '''
        點 : GL_POINTS
        線 : GL_LINES
        連續線 : GL_LINE_STRIP
        封閉線 : GL_LINE_LOOP
        獨立三角形 : GL_TRIANGLES
        連續三角形 : GL_TRIANGLE_STRIP
        三角形扇面 : GL_TRIANGLE_FAN
        4 GL_QUADS
        '''

        glPushMatrix()

        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0) #RGB
        #world Xaxis
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(10.0,0.0,0.0)
        #world Yaxis
        glColor3f(0.0,1.0,0.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,10.0,0.0)
        #world Zaxis
        glColor3f(0.0,0.0,1.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,0.0,10.0)
        glEnd()

        glPopMatrix()

        if not self.query_gl.empty():
            data = self.query_gl.get()
            self.src = data[0]
            self.p1 = data[1]
            self.p2 = data[2] 
            self.p3 = data[3]
            self.p4 = data[4]
            self.p5 = data[5]
            self.end = data[6]

        self.draw_puma()

        self.draw_ground()
        if self.window:
            glutSwapBuffers()
        self.ts.append(time.time())

        if len(self.ts) >= 2:
            dt = self.ts[-1] - self.ts[-2]
    

    def mouseWheel(self,button,dir,x,y):
        if (dir > 0):
            #Zoom in 
            self.cammat = self.cammat @ self.TransXYZ(dz = -0.5)
        else:
            self.cammat = self.cammat @ self.TransXYZ(dz = 0.5)

        
    def keyboard_envent(self,c,x,y):
        print("enter" ,ord(c) ,x ,y)

        if c == 27:
            print("exit")
        if ord(c) == ord('a'):
            self.cammat = self.cammat @ self.TransXYZ(dx = -2.5)
        if ord(c) == ord('d'):
            self.cammat = self.cammat @ self.TransXYZ(dx = 2.5) 
        if ord(c) == ord('w'):
            self.cammat = self.TransXYZ(dz = 2.5) @ self.cammat
        if ord(c) == ord('s'):
            self.cammat = self.TransXYZ(dz = -2.5) @ self.cammat

    def mouse_event(self,button,state,x,y):
        if button == GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN):
                print("LB_DOWN x:" ,x ,"y",y)
        elif button == GLUT_RIGHT_BUTTON:
            if (state == GLUT_DOWN):
                print("RB_DOWN x:" ,x ,"y",y)
        elif button == GLUT_MIDDLE_BUTTON:
            if (state == GLUT_DOWN):
                print("MB_DOWN x:" ,x ,"y",y)

    def draw_tag(self,wTq):
        wTq[:3,3] = wTq[:3,3] * 0.1
        q = self.draw_crood(wTq,5)

    def draw_puma(self):
        src = self.draw_crood(self.src)
        # p1 = self.draw_crood(self.p1)
        # p2 = self.draw_crood(self.p2)
        p3 = self.draw_crood(self.p3)
        # p4 = self.draw_crood(self.p4)
        # p5 = self.draw_crood(self.p5)
        end = self.draw_crood(self.end)

        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0)
        #src->p1 p1->p2,p2->p3
        glVertex3f(0,0,0)
        glVertex3f(src[0],src[1],src[2])
        # glVertex3f(p1[0],p1[1],p1[2])
        # glVertex3f(p1[0],p1[1],p1[2])
        # glVertex3f(p2[0],p2[1],p2[2])
        # glVertex3f(p2[0],p2[1],p2[2])
        glColor3f(0.0,1.0,0.0)
        glVertex3f(src[0],src[1],src[2])
        glVertex3f(p3[0],p3[1],p3[2])

        glColor3f(1.0,0.0,0.0)
        #p3->p4 ,p4->p5
        # glVertex3f(p3[0],p3[1],p3[2])
        # glVertex3f(p4[0],p3[1],p3[2])

        glColor3f(1.0,1.0,0.0)
        # glVertex3f(p4[0],p3[1],p3[2])
        #glVertex3f(p4[0],p4[1],p4[2])
        # glVertex3f(p5[0],p5[1],p5[2])


        #p5 t0 end
        glVertex3f(p3[0],p3[1],p3[2]) 
        # #glVertex3f(p4[0],p4[1],p4[2])
        # # glVertex3f(p5[0],p5[1],p5[2])
        glVertex3f(end[0],end[1],end[2])
        glEnd()

    def draw_crood(self,crood,scale = 3):
        glBegin(GL_LINES)
        pos = crood[:3,3]
        xpos = pos + crood[:3,0] * scale
        ypos = pos + crood[:3,1] * scale
        zpos = pos + crood[:3,2] * scale

        glColor3f(1.0,0.0,0.0) #RBG
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(xpos[0],xpos[1],xpos[2])

        glColor3f(0.0,1.0,0.0)
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(ypos[0],ypos[1],ypos[2])

        glColor3f(0.0,0.0,1.0)
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(zpos[0],zpos[1],zpos[2])
        glEnd()

        return pos
    
    def getpos(self,coord):
        pos = coord[:3,3]

        return pos
    
    def draw_ground(self):
        glColor3f(1.0,1.0,1.0)
        glBegin(GL_LINES)
        for i in range(21):
            data = -100 + i * 10
            glVertex3f(data,100,0)
            glVertex3f(data,-100,0)

            glVertex3f(100,data,0)
            glVertex3f(-100,data,0)
        glEnd() 


class PumaAPP(QDialog):
    def __init__(self):
        super().__init__()

        self.queuy_gl = queue.Queue()
        self.queuy_gait = queue.Queue()
        self.puma = Puma()

        self.view3D = OpenGL_widget(self.queuy_gl,self.puma)

        self.Workready()
        self.view3D.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)# period, in milliseconds

        self.gamethread = None

    def Init(self):
        degs = self.puma.InitPos()

        self.Draw()

    def Workready(self):
        self.puma.ResetPos()
        self.Draw()
    
    def Draw(self):
        data = [self.puma.src,
                self.puma.p1,
                self.puma.p2,
                self.puma.p3,
                self.puma.p4,
                self.puma.p5,
                self.puma.end
            ]

        self.queuy_gl.put(data)

if __name__ =='__main__':
    # q = []
    # test = openGL_Widget(q)
    # test.start()
    # while(1):
    #     try:
    #         time.sleep(1)
    #     except KeyboardInterrupt:
    #         break

    app = QApplication(sys.argv)
    w = PumaAPP()
    w.show()
    app.exec()
    print('exit form')
