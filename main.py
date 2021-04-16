import streamlit as st
import os
import shutil
from fastai.vision import *
from PIL import Image
import numpy as np
import requests

st.set_page_config(page_title="healthy unhealthy leaves",page_icon='',layout="centered",initial_sidebar_state="auto")
st.title('Please upload an image of a leaf')

@st.cache
def load_model():
	save_dest = 'model'
	if not os.path.exists("model/export.pkl"):
		with st.spinner("Loading model... this may take awhile! \n Don't stop it!"):
			if not os.path.exists('model'):
				os.mkdir(save_dest)
			from urllib.request import urlretrieve
			url = 'https://www.dropbox.com/s/60mgysy6hu0d6ms/export.pkl?raw=1'
			urlretrieve(url, 'model/export.pkl')
load_model()

uploaded_file = st.file_uploader('Choose an image')
if uploaded_file is not None:
	model_inf = load_learner(r'model')
	pred = model_inf.predict(open_image(uploaded_file))
	st.write('The leaf is **'+str(pred[0]).upper()+'**')
	st.write('Certainty: '+str(float(pred[-1][pred[-2]])*100)+'%')
	bytes_data = uploaded_file.getvalue()
	st.image(bytes_data, caption = 'Uploaded image', width = 200)


else:
	img = Image.open("images/image_ex.png")
	st.write('Example of images used to train this model')
	st.image(img,width = 200)

st.write('OBS: The cleaner the background the better')