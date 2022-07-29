import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


ops=st.sidebar.radio("MENU", options=("HOME", "PROGRAM", "ABOUT"))

if ops=="HOME":
    st.title("DETEKSI KEMATANGAN BUAH TOMAT")
    st.write("Mengklasifikasi gambar tomat dengan kategori matang, setengah matang, dan mentah.")


    st.caption('Contoh gambar tomat:')
    col1, col2, col3= st.columns(3)
    with col1:
        st.header('TOMAT MATANG')
        st.image('MATANG.jpg')
    with col2:
        st.header('TOMAT SETENGAH MATANG')
        st.image('SETENGAH MATANG.jpg')
    with col3:
        st.header('TOMAT MENTAh')
        st.image('MENTAH.jpg') 


elif ops=="PROGRAM":
    st.title('PROGRAM DETEKSI KEMATANGAN BUAH TOMAT')
    st.image('laptop-3706810__340.webp',caption=('Click Browse Files'))
    
    file = st.file_uploader("Pastikan Gambar Bertype JPG/JPEG", type=['jpg','jpeg'])

    def predict_stage(image_data,model):
        size =(200,200)
        image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
        image_array = np.array(image)
        normalized_image_array =(image_array.astype(np.float32) / 127.0)
        data = np.ndarray(shape=(1, 200, 200, 3),dtype=np.float32)
        data[0] = normalized_image_array
        preds = ""
        prediction = model.predict(data)
        if np.argmax(prediction)==0:
            st.write(f"Tomat Matang")
        elif np.argmax(prediction)==1:
            st.write(f"Tomat Mentah")
        else :
            st.write(f"Tomat Setengah Matang")
    
        return prediction

    if file is None :
        st.text("MASUKAN GAMBAR TOMAT ANDA")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        model = tf.keras.models.load_model('savetugas.h5')
        Generate_pred = st.button("Memprediksi tingkat kematangan.....")
        if Generate_pred:
            prediction = predict_stage(image,model)
            st.text("Probalitas (0: Tomat Matang, 1: Tomat Mentah, 2: Tomat Setengah Matang)")
            st.write(prediction)
            
else:
    st.title('KELOMPOK 3')
    st.caption('SMK N 5 Samarinda')

    col1, col2 =st.columns(2)

    with col1:
        st.image('photo_2022-04-11_08-19-49.jpg')
        st.write('Ifnu Umar')
        st.caption('Program & Web')
    
    with col2:
        st.image('photo_2022-07-09_21-04-25.jpg')
        st.write('Maruto Nugroho')
        st.caption('Dataset')