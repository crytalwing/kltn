import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from keras_self_attention import SeqSelfAttention
import streamlit as st





def main():
    PATH_IMG = 'images/law_fasttext_tvpl_250_128_10_0.2_0.1_31326_104457.61.png'
    st.title('Biểu đồ traning model')

    # Tải hình ảnh từ đường dẫn cục bộ
    image_path = "example_image.jpg"
    image = open(PATH_IMG, 'rb').read()

    # Hiển thị hình ảnh
    st.image(image, caption='Hình ảnh minh họa', use_column_width=True)

if __name__ == "__main__":
    main()
