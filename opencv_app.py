import cv2 as cv # for the image processing
import numpy as np # for the image processing
import streamlit as st # for the UI
from PIL import Image # for the image processing


# function to convert the image to grayscale
def convert_to_grayscale(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return gray

# define function to detect edges
def detect_edges(img):
    edges=cv.Canny(img,100,200)
    return edges

# define function to detect faces
def detect_faces(img):
    face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),6)
    return img


# set title of web app
st.title("OpenCV App for Image Processing")

# add a button to upload the image file from user
uploaded_file=st.file_uploader("Choose an image file",type=['jpg','png','jpeg','webp'])

if uploaded_file is not None:
    file_bytes=np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    image=cv.imdecode(file_bytes,1)

    # Display original image
    st.image(image,channels="BGR",caption="Original Image")

    # when grayscale is clicked, convert image to grayscale
    if st.button("Convert to Grayscale"):
        gray_image=convert_to_grayscale(image)
        st.image(gray_image,channels="GRAY",caption="Grayscale Image")

    # when detect edges is clicked, detect edges in the image
    if st.button("Detect Edges"):
        edges=detect_edges(image)
        st.image(edges,channels="GRAY",caption="Edge Detected Image")

    # when detect faces is clicked, detect faces in the image
    if st.button("Detect Faces"):
        faces=detect_faces(image)
        st.image(faces,channels="BGR",caption="Face Detected Image")
    