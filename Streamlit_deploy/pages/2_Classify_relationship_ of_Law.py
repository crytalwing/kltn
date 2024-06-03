import re
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import libs.Extract_entity as extract_entity

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from keras_self_attention import SeqSelfAttention
import streamlit as st

def tokenization(data, max_length):
    Text_col = data
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(Text_col)
    words_to_index = tokenizer.word_index
    vocab_size = len(words_to_index) + 1

    # chuyển đổi dữ liệu văn bản thành các chuỗi số
    tokenizer_seq = tokenizer.texts_to_sequences(Text_col)

    # Đảm bảo mỗi sequece có cùng độ dài
    # 'post' có nghĩa là thêm padding vào cuối mỗi sequence
    # 'pre' có nghĩa là thêm padding vào đầu mỗi sequence
    tokenizer_pad = pad_sequences(tokenizer_seq, padding='post', maxlen=max_length)

    return tokenizer_pad, vocab_size, words_to_index

def entity_Extraction(A, sentence):
    
    regex_patterns = [r'([^;.<()>:]*)(Hiến_pháp) (?:(nước cộng_hoà xã_hội chủ_nghĩa việt_nam)|(\d+))([^;.<()>:]*)', "Bộ_luật", "Luật", "Pháp_lệnh", "Lệnh", "Quyết_định", 
                      "Nghị_định", "Nghị_quyết", "Nghị_quyết liên_tịch", "Thông_tư", "Thông_tư liên_tịch", "Chỉ_thị"]

    list_B = {
        'Thực thể tham chiếu': [],
        'Nội dung trước thực thể được tham chiếu': [],
        'Thực thể được tham chiếu': [],
        'Nội dung sau thực thể được tham chiếu': []
    }
    for key in regex_patterns:
        if key == regex_patterns[0]:
            matches = re.findall(key, sentence)
        elif (key == 'Nghị_định' or key == 'Thông_tư') and 'liên_tịch' in sentence[sentence.find(key) + 9 : sentence.find(key) + 20]:
                continue
        else:
            pattern = r'([^;.<()>:]*)'+rf'({key})' + r' ((?:(?!ngày|số|này|tại|theo|thì|và|[A-ZĐ])\w+[ _])+)?(số)?\s?(\d+?[\w]+(?:(?:/|-)+(?:\d|\w)+)+)?\s?(ngày \d{1,2}(?: tháng |[-]|[/])\d{1,2}(?: năm |[-]|[/])\d{4})?([^;.<()>:]*)'
            matches = re.findall(pattern, sentence)

        list_entity = list(set(matches))
        if list_entity:     
            for entity in list_entity:
                start_entity = entity[0]
                end_entity = entity[-1]
                entity_tuple = tuple(filter(lambda x: x != '', entity[1:-1]))
                B = ' '. join(entity_tuple).replace('  ', ' ')
                
                if len(entity_tuple) < 2 or B in A:
                    continue
                
                list_B['Thực thể tham chiếu'].append(A)
                list_B['Nội dung trước thực thể được tham chiếu'].append(start_entity)
                list_B['Thực thể được tham chiếu'].append(B)
                list_B['Nội dung sau thực thể được tham chiếu'].append(end_entity)


    df = pd.DataFrame(list_B) 

    return df

def main(input_A,input_B):

    PATH_MODEL = 'models/classify_law_rel_tvpl_fasttext_250_128_10_0.2_0.1_31326_104457.61.h5'
    max_length = 250

    with tf.keras.utils.custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
        # Load mô hình
        loaded_model = load_model(PATH_MODEL)
    
    # Xử lý những trường hợp lí tự dính nhau
    extraction = extract_entity.tvpl_function()
    A_processing = extraction.data_processing(input_A)
    B_processing = extraction.data_processing(input_B)

    X = entity_Extraction(A_processing, B_processing)
    X['Text'] = X.apply(lambda d: f"{d['Thực thể tham chiếu']} {d['Nội dung trước thực thể được tham chiếu']} {d['Thực thể được tham chiếu']} {d['Nội dung sau thực thể được tham chiếu']}".strip(), axis=1)
    
    data_tokenizer_pad_new, vocab_size_new, words_to_index_new = tokenization(X['Text'], max_length)

    label = ['BTT', 'DC', 'DHD', 'DSD', 'CC', 'HHL', 'None']


    # Dự đoán trên dữ liệu mới (X_new)
    predictions = loaded_model.predict(data_tokenizer_pad_new)


    # Mã hóa label
    label_encode = LabelEncoder()
    label_encode = label_encode.fit(label)
    y_pred_original = label_encode.inverse_transform(np.argmax(predictions, axis=1))

    

    if y_pred_original == 'BTT':
        y_pred_original_text = 'Bị thay thế'
    elif y_pred_original == 'DC':
        y_pred_original_text = 'Dẫn chiếu'
    elif y_pred_original == 'DHD':
        y_pred_original_text = 'Được hướng dẫn'
    elif y_pred_original == 'DSD':
        y_pred_original_text = 'Được sửa đổi hoặc bổ sung'
    elif y_pred_original == 'CC':
        y_pred_original_text = 'Căn cứ'
    elif y_pred_original == 'HHL':
        y_pred_original_text = 'Hết hiệu lực'
    else:
        y_pred_original_text = 'Không có quan hệ với thực thể tham chiếu'

    output = pd.DataFrame({'Thực thể tham chiếu': X['Thực thể tham chiếu'], 'Quan hệ': [y_pred_original_text], 'Thực thể được tham chiếu': X['Thực thể được tham chiếu']})

    output['Thực thể được tham chiếu'] = output['Thực thể được tham chiếu'].astype(str).str.pad(width=155, side='right')
    
    st.write(output, heigth=300)


if __name__ == "__main__":

    st.write("# Phân loại quan hệ tham chiếu")

    input_A = st.text_input("Nhập đối tượng tham chiếu", "")
    input_B = st.text_input("Nhập đoạn văn chứa đối tượng được tham chiếu", "")

    if(input_B != ""):
        main(input_A, input_B)
