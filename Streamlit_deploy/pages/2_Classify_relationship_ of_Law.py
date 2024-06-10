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

def load_dicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Chuyển đổi mã kí tự 1252 sang UTF-8
def covert_unicode(text):
    dicchar = load_dicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], text)

def text_preprocessing(document):
    # Chuyển đổi mã kí tự 1252 sang UTF-8
    document['Text'] = document['Text'].apply(covert_unicode)
    # Đưa về dạng chữ thường
    document['Text'] = document['Text'].str.lower()
    # xóa các ký tự không cần thiết
    document['Text'] = document['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    document = document['Text'].apply(lambda y: re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',str(y)))
    return document

def entity_Extraction(A, sentence):
    
    regex_patterns = [r'([^;.<()>:]*)(Hiến_pháp) (?:(nước cộng_hoà xã_hội chủ_nghĩa việt_nam)|(\d+))([^;.<()>:]*)', 
                      "Bộ_luật", 
                      "Luật", 
                      "Pháp_lệnh", 
                      "Lệnh", 
                      "Quyết_định", 
                      "Nghị_định", 
                      "Nghị_quyết", 
                      "Nghị_quyết liên_tịch", 
                      "Thông_tư", 
                      "Thông_tư liên_tịch", "Chỉ_thị"]

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
            pattern = rf'({key})' + r' ((?:(?!ngày|tháng|năm|số)\w+[\s_,]+)+)?(_số|số)?\s?(\d+?[\w]+(?:(?:/|-)+(?:\d|\w)+)+)?\s?((?:năm |tháng |ngày |)(?:\d{4}|\d{1,2}){1})?((?: năm | tháng |[-]|[/])(?:\d{4}|\d{1,2}))?((?: năm |[-]|[/])\d{4})?'
            matches = re.findall(pattern, sentence)
        
        list_entity = list(set(matches))
        if list_entity:     
            for entity in list_entity:
                entity = list(entity)
                entity[4] = ''.join(entity[4:])
                entity = [item for item in entity[:5] if item]

                B = ' '.join(entity).replace('  ', ' ')
                if len(entity) < 2 or B in A:
                    continue
                pattern_i = r'([\w+\s_^;.<()>:]+)' + rf'{B}' + r'([\w+\s_^;.<()>:]+)'
                match_i = re.findall(pattern_i, sentence)
                for match in match_i:
                    list_B['Thực thể tham chiếu'].append(A)
                    list_B['Nội dung trước thực thể được tham chiếu'].append(match[0])
                    list_B['Thực thể được tham chiếu'].append(B)
                    list_B['Nội dung sau thực thể được tham chiếu'].append(match[1])

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
    X['Text'] = text_preprocessing(X)

    data_tokenizer_pad_new, vocab_size_new, words_to_index_new = tokenization(X['Text'], max_length)

    label = ['BTT', 'DC', 'DHD', 'DSD', 'CC', 'HHL', 'None']



    # Dự đoán trên dữ liệu mới (X_new)
    predictions = loaded_model.predict(data_tokenizer_pad_new)


    # Mã hóa label
    label_encode = LabelEncoder()
    label_encode = label_encode.fit(label)
    y_pred_original = label_encode.inverse_transform(np.argmax(predictions, axis=1))

    y_pred_original_lst = []

    for i in y_pred_original:
        if i == 'BTT':
            y_pred_original_text = 'Bị thay thế'
        elif i == 'DC':
            y_pred_original_text = 'Dẫn chiếu'
        elif i == 'DHD':
            y_pred_original_text = 'Được hướng dẫn'
        elif i == 'DSD':
            y_pred_original_text = 'Được sửa đổi hoặc bổ sung'
        elif i == 'CC':
            y_pred_original_text = 'Căn cứ'
        elif i == 'HHL':
            y_pred_original_text = 'Hết hiệu lực'
        else:
            y_pred_original_text = 'Không có quan hệ với thực thể tham chiếu'
        
        y_pred_original_lst.append(y_pred_original_text)

    st.markdown(f"<hr><br> <u>Văn bản:</u> <h4>{A_processing}</h4> <br>", unsafe_allow_html=True)

    output = pd.DataFrame({'Quan hệ': y_pred_original_lst, 'Thực thể được tham chiếu': X['Thực thể được tham chiếu']})

    output['Thực thể được tham chiếu'] = output['Thực thể được tham chiếu'].astype(str).str.pad(width=155, side='right')
    
    st.dataframe(output, width=700, height=400)


if __name__ == "__main__":

    

    st.write("# Phân loại quan hệ tham chiếu")

    tabs_font_css = """
    <style>
    div[class*="stTextArea"] label {
        font-size: 30px;
        color: red;
    }

    div[class*="stTextInput"] label {
        font-size: 30px;
        color: cyan;
    }
    </style>
    """

    st.write(tabs_font_css, unsafe_allow_html=True)
    
    input_A = st.text_input("Nhập đối tượng tham chiếu", "")
    input_B = st.text_area("Nhập đoạn văn chứa đối tượng được tham chiếu", "")

    if(input_B != ""):
        main(input_A, input_B)
