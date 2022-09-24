import streamlit as st
import numpy as np
import cv2
import pandas as pd
import os
from skimage.filters import roberts ,sobel,scharr,prewitt
from scipy import ndimage as nd
import pickle
import torch
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred) 
def get_model():
    with open(r"C:\Users\AMIT PAREEK\Documents\mtech 2nd sem\MTP\feature_model_using_rf", 'rb') as file:
        model1= pickle.load(file)
    
        model2=load_model(r"C:\Users\AMIT PAREEK\Downloads\unet_underwater_splat_mask2_256.h5",custom_objects={"jacard_coef_loss":jacard_coef_loss,"jacard_coef":jacard_coef})
        
        
    return model1,model2
def get_classification_model():
    file=r"C:\Users\AMIT PAREEK\Downloads\classes_classification_googlenet_256.h5"
    model3=torch.load(file,map_location=torch.device('cpu'))
    
    return model3

def loader_image(image_file):
    train_img = load_img(image_file,target_size=(256,256))
    train_img=img_to_array(train_img)
    train_img1=cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    train_img1=np.uint8(train_img1)
    train_img=np.uint8(train_img)
    return train_img1,train_img
    
def feature_extraction(img):
    w,h=img.shape
    df=pd.DataFrame()
    img2=img.reshape(-1)
    df["original_image"]=img2
    num=1 #to count numbers up in order to give gabor feature a label in data frame 
    kernels=[]
    for theta in range(2):
        theta=theat=theta/4.*np.pi
        for sigma in (1,3):
            for lamda in np.arange(0,np.pi,np.pi/4):
                for gamma in (.05,.5):
                    gabor_label="gabor"+str(num) #label gabor colums AS GABOR1 ,GABOR2,GABOR3
                    ksize=3
                    kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg=cv2.filter2D(img2,cv2.CV_8UC3,kernel)
                    filtered_img=fimg.reshape(-1)
                    df[gabor_label]=filtered_img
#                     print(gabor_label,": theta=",theta,": sigma=",sigma,":lamda=",lamda,":gamma=",gamma)
                    num+=1
                    
#     canny                
    edges=cv2.Canny(img,70,150)
    edges1=edges.reshape(-1)
    df["Canny edge"]=edges1    
    
    edge_roberts=roberts(img)
    edge_robert1=edge_roberts.reshape(-1)
    df["roberts"]=edge_robert1
     
    edge_scharr=scharr(img)
    edge_scharr1=edge_scharr.reshape(-1)
    df["scharr"]=edge_scharr1
    
    edge_prewitt=prewitt(img)
    edge_prewitt=edge_prewitt.reshape(-1)
    df["prewitt"]=edge_prewitt
    
    edge_sobel=sobel(img)
    edge_sobel1=edge_sobel.reshape(-1)
    df["sobel"]=edge_sobel1
    
    median_img2=nd.median_filter(img,size=5)
    median_img3=median_img2.reshape(-1)
    df["median 5"]=median_img3
    
    median_img=nd.median_filter(img,size=3)
    median_img1=median_img.reshape(-1)
    df["median 3"]=median_img1
    
    gaussian_img4=nd.gaussian_filter(img,sigma=7)
    gaussian_img5=gaussian_img4.reshape(-1)
    df["gaussian_s7"]=gaussian_img5
    
    gaussian_img2=nd.gaussian_filter(img,sigma=5)
    gaussian_img3=gaussian_img2.reshape(-1)
    df["gaussian_s5"]=gaussian_img3

    gaussian_img=nd.gaussian_filter(img,sigma=3)
    gaussian_img1=gaussian_img.reshape(-1)
    df["gaussian_s3"]=gaussian_img1
    
    variance_img=nd.generic_filter(img,np.var,size=3)
    variance_img1=variance_img.reshape(-1)
    df["variance s3"]=variance_img1
    
    return df
    
def main():
    st.title("Image segmentation")
    menu=["Home","about"]
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice=="Home":
        st.subheader("Home")
        image_file=st.file_uploader("Uplaod image",type=["tif","PNG","JPEG","png","jpeg"])
        
        if image_file is not None:
            # s=str(image_file)
            # z=s.split("/")
            # button4=st.button(f"{image_file.name}")
            img,img2=loader_image(image_file)
            st.image(img)
            button1=st.button("segmetaion")
            if button1:
                data_frame=feature_extraction(img)
                model1,model2=get_model()
                output1=model1.predict(data_frame)
                segmented=output1.reshape((img.shape))
                
                z=[]
                z.append(img2)
                z=np.array(z)
                output2=model2.predict(z)
                col1,col2=st.columns(2)
                with col1:
                    st.text("Output of ML model")
                    st.image(segmented)
                with col2:
                    st.text("Output of DL model")
                    st.image(output2)
                
            button2=st.button("classification")
            if button2:
                                
                classes=np.array(pd.read_csv(r"C:\Users\AMIT PAREEK\Downloads\label (1).csv"))
                idx_to_class={}
                class_to_idx={}
                counter=0
                for i in classes:
                    class_to_idx[i[0]]=counter
                    idx_to_class[counter]=i[0]
                    counter+=1

                
                model3=get_classification_model()
                device=torch.device("cpu")
                model3.to(device)
                img2=np.array(img2)
                img2=np.uint8(img2)
                transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(.2623, .1386)])
                img2= transform(img2)[None,]
                
                out = model3(img2)
                clsidx = torch.argmax(out)
                z=int(clsidx)
                classes=idx_to_class[z]
                st.subheader(f"predicted class is {classes}")
                
                
                
                
            
            
            
            
            
            
             
    else:
        st.subheader("About")
        

if __name__=="__main__":
    main()
    