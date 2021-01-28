# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:37:58 2019

@author: Bourne
"""

from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir,width=512,height=512):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BICUBIC)    # BICUBIC
#        new_img=img.resize((width,height),Image.BILNEAR)     BILNEAR METHOD 
#        new_img=img.resize((width,height),Image.NEAREST)     NEARST METHOD 
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
        
for jpgfile in glob.glob("C:/Users/Bourne/Desktop/cropped/target/*.jpg"):  ## TARGET IMAGE FILE LOCATION 
    convertjpg(jpgfile,"C:/Users/Bourne/Desktop/NORMAL3")                  ## SAVE RESULT IMAGE FILE LOCATION 
