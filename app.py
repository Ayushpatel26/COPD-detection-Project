# write a code for audio classification using model.h5 and  streamlit in python

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
# Load various imports 
from datetime import datetime
from os import listdir
from os.path import isfile, join

import librosa
import librosa.display

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

st.title('COPD Classification ')
st.header('By Ayush,Asheesh and Rahul')
# Load the model
model = load_model('cnn.h5')
mypath = "C:/Users/Ayush Patel/Desktop/copd/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))] 
#prediction
max_pad_len = 862 
le = LabelEncoder()

p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file) 

p_diag = pd.read_csv("C:/Users/Ayush Patel/Desktop/copd/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None) # patient diagnosis file
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file]) # labels for audio files


c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
def Prediction(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    features = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    features = np.array(features)
    features = np.expand_dims(features,axis=2)
    prediction = model.predict(features)
    
    predicted_class = c_names[np.argmax(prediction[0])]
    print("ANN has predicted the class as  --> ", predicted_class[0])
    return predicted_class

uploaded_file = st.file_uploader("Choose an Audio file", type=['wav','mp3'], accept_multiple_files=False)
st.write("Audio name:", uploaded_file)

if st.button("Predict"):
    sample_rate,predicted_class = Prediction(uploaded_file)
    st.write(predicted_class)