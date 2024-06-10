import re
import json
import string
import enchant
from pyvi import ViTokenizer

class tvpl_function:
    def __init__(self):
        self.pattern = 'aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơớờởỡợuùủũúụưừửữứựyỳỷỹýỵđ'

        self.words_en = enchant.Dict("en_US")
        
        self.specialCharset = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'

        self.words = []
        with open("libs/words.txt", "r", encoding='utf-8') as file:
            for f in file:
                word = json.loads(f)
                self.words.append(word["text"])

        self.syllables = []
        with open("libs/syllable.txt", "r", encoding="utf-8") as file:
            for line in file:
                self.syllables.append(line.strip())
                
        self.mapping={
            'óa':'oá', 'òa':'oà', 'ỏa':'oả', 'õa':'oã', 'ọa':'oạ',
            'óe':'oé', 'òe':'oè', 'ỏe':'oẻ', 'õe':'oẽ', 'ọe':'oẹ',
            'úy':'uý', 'ùy':'uỳ', 'ủy':'uỷ', 'ũy':'uỹ', 'ụy':'uỵ',
            'úâ':'uấ', 'ùâ':'uầ', 'ủâ':'uẩ', 'ũâ':'uẫ', 'ụâ':'uậ',
            'úe':'ué', 'ùe':'uè', 'ủe':'uẻ', 'ũe':'uẽ', 'ụe':'uẹ',
            'úê':'uế', 'ùê':'uề', 'ủê':'uể', 'ũê':'uễ', 'ụê':'uệ',
            'úơ':'uớ', 'ùơ':'uờ', 'ủơ':'uở', 'ũơ':'uỡ', 'ụơ':'uợ',
            'úô':'uố', 'ùô':'uồ', 'ủô':'uổ', 'ũô':'uỗ', 'ụô':'uộ',
            'iá':'ía', 'ià':'ìa', 'iả':'ỉa', 'iã':'ĩa', 'iạ':'ịa',
            'yá':'ýa', 'yà':'ỳa', 'yả':'ỷa', 'yã':'ỹa', 'yạ':'ỵa',
            'uá':'úa', 'uà':'ùa', 'uả':'ủa', 'uã':'ũa', 'uạ':'ụa',
            'ưá':'ứa', 'ưà':'ừa', 'ưả':'ửa', 'ưã':'ữa', 'ưạ':'ựa',
            'ứơ':'ướ', 'ừơ':'ườ', 'ửơ':'ưở', 'ữơ':'ưỡ', 'ựơ':'ượ',
            'íê':'iế', 'ìê':'iề', 'ỉê':'iể', 'ĩê':'iễ', 'ịê':'iệ',
            'ýê':'yế', 'ỳê':'yề', 'ỷê':'yể', 'ỹê':'yễ', 'ỵê':'yệ',
            'uí':'úi', 'uì':'ùi', 'uỉ':'ủi', 'uĩ':'ũi', 'uị':'ụi',
            'aó':'áo', 'aò':'ào', 'aỏ':'ảo', 'aõ':'ão', 'aọ':'ạo',
            'qúa':'quá', 'qùa':'quà', 'qủa':'quả', 'qũa':'quã', 'qụa': 'quạ',
            'Qúa':'Quá', 'Qùa':'Quà', 'Qủa':'Quả', 'Qũa':'Quã', 'Qụa': 'Quạ', 
            'gía':'giá', 'gìa':'già', 'gỉa':'giả', 'gĩa':'giã', 'gịa': 'giạ',
            'Gía':'Giá', 'Gìa':'Già', 'Gỉa':'Giả', 'Gĩa':'Giã', 'Gịa': 'Giạ',
        }
        
        self.type_documents =['Hiến_pháp', 'Bộ_luật', 'Luật', 'Pháp_lệnh', 'Lệnh', 'Nghị_quyết', 'Nghị_quyết liên_tịch', 'Nghị_định', 'Quyết_định', 'Thông_tư', 'Thông_tư liên_tịch', 'Chỉ_thị']
        

    # chuẩn hóa bảng mã tiếng việt
    # Tạo ra từ điện bộ kí tự
    def load_DictChar(self):
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic
    
    # Chuyển đổi mã kí tự 1252 sang UTF-8
    def covert_unicode(self, txt):
        dicchar = self.load_DictChar()
        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()], txt) 

    # xóa phần không cần thiết trong văn bản
    def removeUnnecessaryPart(self, row):
        for i, line in enumerate(row.splitlines()):
            if re.match(r"^(PHẦN PHỤ LỤC|DANH MỤC PHỤ LỤC)", line) or re.match(r"^phụ lục", line.lower()) or re.match(r"^(\s|\t)*PHỤ LỤC", line):
                row = '\n'.join(row.splitlines()[:i])
                break     
        
        return row

    #  Tìm và thay thế các ký tự không in được trong text thành khoảng trắng
    def  replace_non_printable_chars(self, row):
        return  ''.join(char if char in string.printable or char.lower() in self.pattern else ' ' for char in row)
                
    def data_processing(self, row):
        row = self.covert_unicode(row)                          # Chuẩn hóa bảng mã kí 
        row = self.removeUnnecessaryPart(row)                   # Loại bỏ phần không cần thiết
        row = self.replace_non_printable_chars(row)             # Thay thế các chuỗi không in được thành khoảng trắng
        
        result_line = []
        for line in row.splitlines():           
            if line.strip():
                line = self.sentence_processing(line)           # Tách âm tiết bị dính nhau
                line = self.split_word(line)                    # Tách câu, tách từ tiếng              
                result_line.append(line)
        
        row = "\n\n".join(result_line)
        row = self.highlight_object(row)

        return row

    # Tách câu, tách từ tiếng 
    def split_word(self, sentence):
        pattern = re.compile(r'\d{1,4}(/|-)\d{1,4}(/|-)?\w{0,7}(-\w{2,4})?|\d{1,4}(/|-)\w{2,7}(-\w{2,4})?')
        result = ''
        i_start = 0
        str_list =  sentence.split()
        
        for i, word in enumerate(sentence.split()):
            if pattern.match(word):
                text = ' '.join(str_list[i_start:i])
                text = ViTokenizer.tokenize(text)
                result += f'{text} {str_list[i]} '
                i_start = i + 1

        if i_start < len(str_list):
            text = ' '.join(str_list[i_start:])
            text = ViTokenizer.tokenize(text)
            result += text

        return result.strip()   

    # Xử lý câu
    def sentence_processing(self, sentence):
        result_sentence = []
        sentence = self.word_processing(sentence)           # Xử lý từ cơ bản

        for word in sentence.split():                  
            # Nếu từ không chứa kí tự, từ có nghĩa, là số ký hiệu hoặc là từ tiếng anh thì bỏ qua
            if (len(word) == 1
                    or any(char in word for char in self.specialCharset)
                    or any(char in word.lower() for char in ["f","j","w","z"])
                    or any(char.isnumeric() for char in word)
                    or self.words_en.check(word) 
                    or self.syllable_check(word) 
                    ):
                result_sentence.append(word)
                continue

            # Từ có nghĩa nhưng có 2 chữ giống nhau liên tiếp (vd: hoaang, luuật)
            word = self.double_char(word)
            if self.syllable_check(word):   # xét từ có nghĩa chưa
                result_sentence.append(word)
                continue

            check = True
            #Kiểm tra cặp từ có nghĩa trong từ điển
            for i in range(1, len(word)-1):
                text = word[:i] + " " + word[i:]
                if self.dictionary_check(text):
                    check = False
                    break
                
            # Nếu không có cặp từ có nghĩa
            if check:
                text = self.split_syllable(word)   # text = text (được tách ra)/ False (không được tách)

            # Xử lý các âm tiết bị lỗi dính nhau (các trường hợp chữ viết tắt và sai chính tả -> False)
            result_sentence.append(text if text else word) # nếu False thì từ được giữ nguyên

        return ' '.join(result_sentence)
    
    # Xử lý từ
    def word_processing(self, sentence):
        result = []
        pattern = re.compile(r'\d{1,4}(/|-)\d{1,4}(/|-)?\w{0,7}(-\w{2,4})?|\d{1,4}(/|-)\w{2,7}(-\w{2,4})?')
        for word in sentence.split():
            word = self.standardize_Tone(word)  # Chuẩn hóa thanh điệu
            word = self.chuan_hoa_y_i(word)     # Chuẩn hóa y/i

            if word.isupper() and not pattern.match(word):
                word = word.capitalize()

            for i in self.specialCharset:
                if ('/' in word or '-' in word) and pattern.match(word) and (i == '/' or i == '-'):
                    continue
                if i in word:
                    word = word.replace(i, f" {i} ")    # Tách các kí tự đặc biệt 
            result.append(word)
        
        return ' '.join(result).replace('  ', ' ').strip()


    # Chuẩn hóa y và i (vd: tỷ lệ -> tỉ lệ, bác sỹ -> bác sĩ)
    def chuan_hoa_y_i(self, word):
        set_yi ={1: 'yýỳỷỹỵ', 2: 'iíìỉĩị'}
        index_yi = None
        index_setyi = None
        for i in range(6):
            if set_yi[1][i] in word or set_yi[2][i] in word:
                index_yi = word.index(set_yi[1][i]) if set_yi[1][i] in word else word.index(set_yi[2][i])
                index_setyi = i
                break

        if index_yi is not None:
            if len(word) == 1 and word in set_yi[2]:
                word = set_yi[1][index_setyi]
            elif (len(word) > 1 and index_yi == len(word) - 1 and word[index_yi] in set_yi[1] 
            and word[:index_yi] in ['b','c','ch','d','đ','g','gh','h','k','kh','l','m','n','ng','ngh','nh','p','ph','r','s','t','th','tr','v','x']):
                word = word[:index_yi] + set_yi[2][index_setyi]

        return word

    # Trả kết quả xử lý tách các âm tiết dính nhau
    def split_syllable(self, syllable):  
        memories = []
        while syllable != '':
            text_len = len(syllable)    # Độ dài từ hiện tại
            memory = []                 # Chứa các từ có nghĩa 
            queue = ''                  # Cập nhật các từ

            for i in range(text_len):
                # Xử lý từ có nghĩa (vd: nóng -> memory = [nó, nón, nóng])
                queue += syllable[i]
                
                # Lọc ra cá kí tự đứng 1 mình, trừ các từ có thể
                if len(queue) == 1 and queue not in ['ả', 'ế', 'ô', 'ố', 'ổ', 'y', 'ý', 'ỷ', 'ủ']:
                    continue

                if self.syllable_check(queue) or self.dictionary_check(queue):
                    memory.append(queue)
            
            # Nếu memory có chứa các từ có nghĩa
            if memory:
                memories.append(memory)
                syllable = syllable[len(memory[-1]):]     # Loại bỏ từ có nghĩa trong chuỗi syllable và xét tiếp theo vòng lặp while
            # Nếu không
            else:
                # Nếu xét cả chuỗi không có từ nào hoặc chỉ 1 từ có nghĩa trong chuỗi -> False
                if memories and len(memories[0]) > 1:
                    # kiểm tra chuỗi từ trước có 2 từ trở lên thì ghép chữ cuối cùng của chuỗi từ đó [-1]  
                    for j in range(len(memories), 0, -1):
                        # Vd: hoặcđiago -> memories = [['cá', 'các', 'cách'],['i']] --- syllable = 'ệp'
                        if len(memories[j-1]) >= 2:
                            syllable = memories[j-1].pop(-1)[-1] + ''.join(c[-1] for c in memories[j:]) + syllable
                            memories = memories[:j]
                            break
                else:
                    return False
        
        return ' '.join(i[-1] for i in memories)

    # kiểm tra từ hoặc cụm từ có trong từ điển
    def dictionary_check(self, text):
        return text.lower() in self.words
    
    # kiểm tra âm tiết có trong từ điển
    def syllable_check(self, syllable):
        return syllable.lower() in self.syllables

    # xóa các kí tự lặp (trừ "o")
    def double_char(self, word):
        result = word[0]  # Khởi tạo result với ký tự đầu tiên của word
        for i in range(1, len(word)):  # Bắt đầu vòng lặp từ chỉ số 1
            if result[-1] == word[i] and word[i] != 'o':
                continue
            result += word[i]

        return result

    # Chuẩn hóa thanh điệu 
    def standardize_Tone(self, word):  
        if word != "gịa" and word != "quốc":
            for key, value in self.mapping.items():
                word = word.replace(key, value)
        return word
    
    # làm đối tượng nổi bật
    def highlight_object(self, sentence):
        for key in self.type_documents:
            if key.lower() in sentence.lower():
                i_start = sentence.lower().index(key.lower())
                i_end = i_start + len(key)
                text = sentence[i_start : i_end]
                sentence = sentence.replace(text, key)    

        
        pattern = r'(\w)_(\w)'
        matches = re.findall(pattern, sentence)
        matches = list(set(matches))
        for match in matches:
            if match[0].islower() and match[1].isupper():
                key_word = '_'.join(match)
                sentence = sentence.replace(key_word, key_word.lower())
                
        if 'pháp lệnh' in sentence.lower():
            start_index = sentence.lower().index('pháp lệnh')
            end_index = start_index + len('pháp lệnh')
            if sentence[start_index-1] == '_':
                sentence = sentence[:start_index - 1] + " " +sentence[start_index:]
            sentence = sentence[:start_index] + 'Pháp_lệnh' + sentence[end_index:]

        return sentence