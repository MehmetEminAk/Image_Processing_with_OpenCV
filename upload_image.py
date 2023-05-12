# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:54:59 2023

@author: Mehmet Emin Ak
"""
#%% Video Okuma
import os
import cv2
import time

video_name = "street_video.mp4"

cap = cv2.VideoCapture(video_name)

print(cap.get(3)) 

if cap.isOpened() == False :
    print("Hata")
    
    
while True :
    ret , frame = cap.read()
    
    if ret == True :
        #Image is successfully scaned
        time.sleep(0.01)
        cv2.imshow("Video" , frame)
    else :
        break
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

#%% Kamera Açma

cap = cv2.VideoCapture(0) # 0 default kameramızdır , bilgisayarda kaç kamera varsa o kadar yazılabilir

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Kameramızın frame genişliği
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Kameramızın frame yüksekliği

#Video kaydet
writer = cv2.VideoWriter("video.mp4",cv2.VideoWriter_fourcc(*"DIVX"),20,(int(width),int(height)))
#Video write four cc çerçeveleri sıkıştırmak için kullanılır
 
while True : 
    ret , frame = cap.read()
    
    cv2.imshow("Video" , frame)
    
    #Save (Writer ı bir frame deposu olarak düşünebilirsin)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q") : break
    
cap.release()
writer.release()
cv2.destroyAllWindows()


#%%  Resimleri yeniden boyutlandırma ve kırpma
import os
import cv2

img = cv2.imread("lenna.png",1)
newImg = cv2.resize(img , (15,15))
print(newImg.shape)
cv2.imshow("Resized",newImg)
cv2.imshow("Normal",img)
imCroped = img[:200,0:300] # 200x300 lük bir resim elde ederiz
cv2.imshow("kirpilmis",imCroped)

#%% Resmin üzerine şekil çizdirme (Nesne tespiti çin) ve resmin üzerine yazı yazdırmak

import cv2
import numpy as np

#resim olşutur
img = np.zeros((512,512,3) , np.uint8) #Siyah bir resim
print(img.shape)
cv2.imshow("Siyah" , img)
cv2.line(img ,(0,0) , (512,0) , (0,255,0) , 3) # Üzerine line ekleme (resim , başlangıç noktası , bitiş noktası ,line' ın !bgr! renk kodu)

cv2.imshow("With Line", img) #Çigili resmi bastır

#Diktörtgen çizdirme
# Parametreler (resim , başlangıç , bitiş , bgr kodu)
cv2.rectangle(img,(0,0) , (256,256),(255,0,0) ,5)
cv2.imshow("Diktortgen" , img)

#Daire çizdirme
cv2.circle(img,(256,256) , 100 , (255,255,255),cv2.FILLED)
cv2.imshow("circle",img)

#Yazı Yazdırma

cv2.putText(img,"Resim\n Nabeerrrrr",(0,0),cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255))

cv2.imshow("Yazili" , img)

#%% Resimlerin birleştirilmesi
import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("Original",img)

#Yan yana concat numpy ın htack i sayesinde yapabiliriz
hor = np.hstack((img,img))
cv2.imshow("Horizantal Concat",hor)

#Dikey olarak concat 

verti = np.vstack((img,img))
cv2.imshow("Vertical",verti)

#%% Perspektif Çarpıtma , Düzeltme
import cv2
import numpy as np


cardImg = cv2.imread("kart.png")
cv2.imshow("Original" , cardImg)

#Çevirmek için transform matrislerini kullanıcaz
#Ama önce en köşe matrislerin konumunu almalıyız

width = 400
height = 400

#Bu ilk pointi paintten aldık
#Çevirmek istediğim kartın 4 bir köşesinin koordinatları
point1 =  np.float32(([203,1],[1,472],[540,150],[338,617]))

#Çevirdikten sonra olması gereken kart köşeleri
point2 = np.float32(([0,0], [0,height] , [width,0] , [width,height]))

matrix = cv2.getPerspectiveTransform(point1,point2)
transformedImg = cv2.warpPerspective(cardImg,matrix,(width,height))
cv2.imshow("Transformed",transformedImg)



#%% Resimleri karıştırarak ter resim elde etme
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")

img2 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)



#Convert colorr BGR to RGB
img2 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)


#Resimleri aynı shape e getirme


img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")

img1 = cv2.resize(img1 , (600,600))
img2 = cv2.resize(img1 , (600,600))

print(img1.shape,img1.shape)

# Karıstırılmıs resim = alpha .* img 1 + beta * img2
#addWeighted fonk sayesinde iki resmi karıştırabiliyoruz
blended = cv2.addWeighted(src1 = img1,alpha = 0.1,src2 = img2,beta = 0.9, gamma = 0.3)
plt.figure()
plt.imshow(blended)
print(blended.shape) 


#%% Görüntülerin eşik değerlerini eşitleme
#(Thresholding) (Eşik değerin üzerindeki pikselleri gösterme , diğerlerini göstermeme)

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg" ,0)
plt.figure()

#Color map i gray tabanlı yapıyoruz ikinci parametre ile silip farkı görebilirsin

plt.imshow(img1 , cmap = "gray")
plt.axis("off")
plt.show()

#Eşikleme
_ , threshedImg = cv2.threshold(img1,thresh = 60,maxval = 255,type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(threshedImg ,cmap = "gray")
plt.show()




#%% Adaptif (Uyarlamalı Thresholding) 
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread("img1.png")

gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

cv2.imshow("deneme",thresh_img)


#%% Blurring (Bulanıklaştırma) Detayı azaltır ve gürültüyü engeller
#OpenCV de 3 çeşit bulanıklatırma vardır
#Ortalama Bulanıklaştırma
#Gauss Bulanıklaştırma
#Medyan Bulanıklaştırma

import cv2
import matplotlib.pyplot as plt
import numpy as np #Gürültü oluşturmak için

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.show()

"""
    Ortalama Bulanıklaştırma (Bir filtre oluyor bu kutu
    filtre her bir pikseli dolaşıyor , ortalamasını 
    alıyor ve gezdiği pikselin değerini değiştiriyor)
    Filtre kutusuna kernel deniyor
"""
dst = cv2.blur(img,ksize = (3,3))
plt.figure() , plt.imshow(dst) , plt.axis("off") , plt.title("Ortalama blur")

"""
    Gausian Blur (Gause noise unu elimine eder)
    
"""
gb = cv2.GaussianBlur(dst, (3,3), 7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("Gauss Blur") , plt.show()


"""
Medyan Blur 
Kernel filtre kutucuğunu düzleştirdiğimizi düşünelim mesela 
3x3 lük kernel vektörel tek boyutlu 9 elemanlı bir dizi olur.
 Bu vektörü sıralayıp ortadaki değeri merkez piksele yazar
"""
mb = cv2.medianBlur(img,3)

plt.figure(),plt.imshow(mb) , plt.axis("off") , plt.title("Medyan Blur"), plt.show()


def gaussinNoise(image):
    
    row , col , ch = image.shape
    
    """
    Bir gaus noise u oluşturmak için ortalama
    değere ve strand .sapma ya ihtiyaç var
    """
    mean = 0
    varyans =  0.05 #Stnadart sapmayı bulmak için çünkü std. sapma varyansın kareköküdür 
   
    sigma = varyans **0.5
    
    #Gaussianın diğer adı normal dağılımdır
    gauss = np.random.normal(mean,sigma, (row,col,ch))

    #Gaus noise u elde ettik yukarıda
    
    gauss = gauss.reshape(row,col,ch)
    
    noisyImage = image + gauss
    return noisyImage

#Gürültüyü eklemek için normalizasyon yapıcaz yani tüm resim piksellerini 0 ile 1 arasına getircez
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) / 255 #Normalizasyon işlemi tam burada oluyor

plt.figure(), plt.imshow(img), plt.show()

gausianNoisyImage = gaussinNoise(img)

plt.figure(), plt.imshow(gausianNoisyImage), plt.show()

#Gausian noisy li image in gürültüsünü giderme


gb2 = cv2.GaussianBlur(gausianNoisyImage, (3,3), 7)
plt.figure(), plt.imshow(gb2), plt.axis("off"), plt.title("With Gauss Blur") , plt.show()


#%% Morfolojik Operasyonlar


import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("datai_team.jpg" , 0)

plt.figure(),plt.imshow(img , cmap = "gray"),plt.axis("off"),plt.title("Original Image")

#   Erozyon (Sınırları küçültüyoruz) (Beyazlıkları azaltıyor)

kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel , iterations = 1)

plt.figure(),plt.imshow(result , cmap = "gray"),plt.axis("off"),plt.title("Iteration Image")

# Erozyonun tersi genişleme (Dilateion)

result2 = cv2.dilate(img , kernel , iterations = 1)
plt.figure(),plt.imshow(result2 , cmap = "gray"),plt.axis("off"),plt.title("Dilation Image")


#Açılma Yöntemi (Beyaz gürültüyü önlemek için kullanılır)
whiteNoise = np.random.randint(0,2,size= img.shape[:2])
whiteNoise = whiteNoise * 255
plt.figure(),plt.imshow(whiteNoise , cmap = "gray"),plt.show()

#Beyaz gürültülü resim
noise_img = whiteNoise + img
plt.figure(),plt.imshow(noise_img , cmap = "gray"),plt.show()

#Beyaz gürültülü resmi açma

opening = cv2.morphologyEx(noise_img.astype(np.float32) , cv2.MORPH_OPEN, kernel)
plt.figure(),plt.imshow(opening , cmap = "gray"),plt.show()

#Black Noise
blackNoise = np.random.randint(0,2,size=img.shape[:2])
blackNoise = blackNoise * -255

plt.figure(),plt.imshow(blackNoise , cmap = "gray"),plt.show()


#gradient (Kenar tespiti yapar) (Bu örnekte yazı üzerindeki)
gradient = cv2.morphologyEx(img , cv2.MORPH_GRADIENT, kernel)
plt.figure(),plt.imshow(gradient , cmap = "gray"),plt.show()


#%% Gradyanlar (Görüntü gradyanı ,görüntüdeki yoğunluk veya renkteki yönlü bir değişikliktir)
#Kenar algılamada kullanılır

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("sudoku.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"),plt.title("Original"),plt.show()

#X eksenindeki gradyanlar
sobelX = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelX, cmap = "gray"), plt.axis("off"),plt.title("x COORDS"),plt.show()

#Y eksenindeki gradyanlar
sobelY = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobelY, cmap = "gray"), plt.axis("off"),plt.title("Y Coords"),plt.show()

# Laplacian Gradyan (Her iki koordinatdan (x ve y) koordinat tespiti yapar)
laplacian = cv2.Laplacian(img,ddepth= cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray"), plt.axis("off"),plt.title("lAPLACİAN"),plt.show()


#%% Histogram 
#Görüntü histogramı, Görüntüdeki ton dağılımının grafiksel bir temsili olarak işlev gören bir histogram türüdür
#Her bir ton değeri için piksel sayısını içerir
#Belirli bir görüntü için histograma bakılarak , ton dağılımı anlaşılabilir

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("red_blue.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.axis("off"),plt.title("Original"),plt.show()




