import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

ops=st.sidebar.radio("MENU", options=("HOME", "PROGRAM", "ABOUT", "CONTACT"))

if ops=="HOME":
    st.title("DETEKSI KEMATANGAN SAYUR TOMAT")
    st.write("Mengklasifikasi gambar (SAYUR) dengan kategori matang, setengah matang, dan mentah.")


    st.caption('Contoh gambar tomat:')
    col1, col2, col3= st.columns(3)
    with col1:
        st.header('TOMAT MATANG')
        st.image('MATANG.jpg')
    with col2:
        st.header('TOMAT SETENGAH MATANG')
        st.image('SETENGAH MATANG.jpg')
    with col3:
        st.header('TOMAT MENTAH')
        st.image('MENTAH.jpg') 


elif ops=="PROGRAM":
    st.title('PROGRAM DETEKSI KEMATANGAN SAYUR TOMAT')
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
            
if ops=="ABOUT": 
    st.title('KELOMPOK 1')
    st.caption('SMK N 5 Samarinda')

    col1, =st.columns(1)
    col2, col3, =st.columns(2)
    col4, col5, =st.columns(2)

    with col1:
        st.image('Grub1.jpeg')
        st.write('Ifnu Umar')
        st.caption('Ketua (Modeling, Program, Dataset)')
    
    with col2:
        st.image('Grub2.jpeg')
        st.write('Muhammad Arif Rahman')
        st.caption('Web Design')
    
    with col3:
        st.image('Grub3.jpeg')
        st.write('Akmal')
        st.caption('........')
    
    with col4:
        st.image('Grub4.jpeg')
        st.write('Rio febrian')
        st.caption('........')
   
    with col5:
        st.image('Grub5.jpeg')
        st.write('Rafael')
        st.caption('........')
    
    st.write("Kami membuat website ini untuk mendescripsikan tomat itu dengan kategori Matang/Setengah Matang/Mentah dengan Mengklasifikasi gambar (BUAH).")

elif ops=="CONTACT":
    st.title('Sosial Media')

    col1, col2, col3, col4, col5 = st.columns( 5)
    col1.write("WhatsApp")
    col2.write("Instagram")
    col3.write("Facebook")
    col4.write("Youtube")
    col5.write("Twitter")
        
    with col1:
        st.image('whatsapp.webp')

        if st.write("[Whatsapp](https://api.whatsapp.com/send/?phone=6289501603099&text=Bang+Save+Saya+%28Nama%29)"):
            st.write()
   
    with col2:
        st.image('Instagram.png')

        if st.write("[Instagram](https://www.instagram.com)"):
            st.write()
   
    with col3:
        st.image('Facebook.png')

        if st.write("[Facebook](https://www.facebook.com)"):
            st.write()
    
    with col4:
        st.image('Youtube.jpg')

        if st.write("[Youtube](https://www.youtube.com)"):
            st.write()

    with col5:
        st.image('Twitter.png')

        if st.write("[Twitter](https://www.twitter.com)"):
            st.write()

        
            