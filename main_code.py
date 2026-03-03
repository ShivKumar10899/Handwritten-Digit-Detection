import torch
import torch.nn as nn
import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer as cam
import av
sk=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
st.title("Real-Time Handwritten Number Detection")
st.subheader('Enter one digit at a time in camera view Write the number big with thiker pen')
device=torch.device("cuda"if torch.cuda.is_available() else"cpu")
print("Using device:",device)
class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer0=nn.Sequential(
            nn.Linear(16*25*25,5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10))
    def forward(self,x):
        x=self.layer(x)
        x=torch.flatten(x,1)
        x=self.layer0(x)
        return x
model=my_model().to(device)
model.load_state_dict(torch.load("Number_Detection.pth"))
model.eval()
def Camara(frame):
    img = frame.to_ndarray(format="bgr24")
    image=cv2.resize(img,(100,100))
    frame=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,frame = cv2.threshold(frame,100,255,cv2.THRESH_BINARY_INV)
    frame=cv2.filter2D(frame,-1,sk)
    frame=frame/255.0
    frame = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output=model(frame)
        pred=output.argmax(dim=1).item()
    cv2.putText(img, f"Predicted Digit is {str(pred)}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")
cam(key="Hand_Written_Digit_Detection", video_frame_callback=Camara,media_stream_constraints={"video": True,"audio": False,}) 
