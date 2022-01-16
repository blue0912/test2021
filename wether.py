#weather1.py
import sys
from PySide2 import QtCore, QtScxml

app = QtCore.QCoreApplication()
el = QtCore.QEventLoop()

sm = QtScxml.QScxmlStateMachine.fromFile('states.scxml')

sm.start()
el.processEvents()

print("SYS> こちらは天気情報案内システムです")

uttdic = {"ask_place":"地名を言ってください",
          "ask_date":"日付を言ってください",
          "ask_type":"情報種別を言ってください"}

current_state = sm.activeStateNames()[0]
print("current_state=",current_state)

sysutt = uttdic[current_state]
print("SYS>",sysutt)

while True:
    text = input("> ")
    sm.submitEvent(text)
    el.processEvents()

    current_state = sm.activeStateNames()[0]
    print("current_state=",current_state)

    if current_state == "tell_info":
        print("天気をお伝えします")
        break
    else:
        sysutt=uttdic[current_state]
        print("SYS>",sysutt)

print("ご利用ありがとうございました")

#weather2.py
import sys
from PySide2 import QtCore, QtScxml

prefs = ['三重','京都','佐賀','兵庫','北海道','千葉','和歌山',
         '埼玉','大分','大阪','奈良','宮城','宮崎','富山','山口',
         '山形','山梨','岐阜','岡山','岩手','島根','広島','徳島',
         '愛媛','愛知','新潟','東京','栃木','沖縄','滋賀','熊本',
         '石川','神奈川','福井','福岡','福島','秋田','群馬','茨城',
         '長崎','長野','青森','静岡','香川','高知','鳥取','鹿児島']

def get_place(text):
    for pref in prefs:
        if pref in text:
            return pref
    return ""

def get_date(text):
    if "今日" in text:
        return "今日"
    elif "明日" in text:
        return "明日"
    else:
        return ""

def get_type(text):
    if "天気" in text:
        return "天気"
    elif "気温" in text:
        return "気温"
    else:
        return ""


app = QtCore.QCoreApplication()
el = QtCore.QEventLoop()

sm = QtScxml.QScxmlStateMachine.fromFile('states.scxml')

sm.start()
el.processEvents()

print("SYS> こちらは天気情報案内システムです")

uttdic = {"ask_place":"地名を言ってください",
          "ask_date":"日付を言ってください",
          "ask_type":"情報種別を言ってください"}

current_state = sm.activeStateNames()[0]
print("current_state=",current_state)

sysutt = uttdic[current_state]
print("SYS>",sysutt)

while True:
    text = input("> ")
    if current_state == "ask_place":
        place = get_place(text)
        if place != "":
            sm.submitEvent("place")
            el.processEvents()
    elif current_state == "ask_date":
        date = get_date(text)
        if date != "":
            sm.submitEvent("date")
            el.processEvents()
    elif current_state == "ask_type":
        _type = get_type(text)
        if _type != "":
            sm.submitEvent("type")
            el.processEvents()

    current_state = sm.activeStateNames()[0]
    print("current_state=",current_state)

    if current_state == "tell_info":
        print("天気をお伝えします")
        break
    else:
        sysutt=uttdic[current_state]
        print("SYS>",sysutt)

print("ご利用ありがとうございました")

#weather3.py
import sys
from PySide2 import QtCore, QtScxml
import requests
import json
from datetime import datetime,timedelta,time

prefs = ['三重','京都','佐賀','兵庫','北海道','千葉','和歌山',
         '埼玉','大分','大阪','奈良','宮城','宮崎','富山','山口',
         '山形','山梨','岐阜','岡山','岩手','島根','広島','徳島',
         '愛媛','愛知','新潟','東京','栃木','沖縄','滋賀','熊本',
         '石川','神奈川','福井','福岡','福島','秋田','群馬','茨城',
         '長崎','長野','青森','静岡','香川','高知','鳥取','鹿児島']

latlondic = {'北海道':(43.06,141.35),'青森':(40.82,140.74),
             '岩手':(39.7,141.15),'宮城':(38.27,140.87),
             '秋田':(39.72,140.47),'山形':(38.24,140.36),
             '福島':(37.75,140.47),'茨城':(36.34,140.45),
             '栃木':(336.57,139.88),'群馬':(36.39,139.06),
             '埼玉':(35.86,139.65),'千葉':(25.61,140.12),
             '東京':(35.69,139.69),'神奈川':(35.45,139.64),
             '新潟':(37.9,139.02),'富山':(36.7,137.21),
             '石川':(36.59,136.63),'福井':(36.07,136.22),
             '山梨':(35.66,138.57),'長野':(36.65,138.18),
             '岐阜':(35.39,136.72),'静岡':(34.98,138.38),
             '愛知':(35.18,136.91),'三重':(34.73,136.51),
             '滋賀':(35.0,135.87),'京都':(35.02,135.76),
             '大阪':(34.69,135.52),'兵庫':(34.69,135.18),
             '奈良':(34.69,135.83),'和歌山':(34.23,135.17),
             '鳥取':(35.5,134.24),'島根':(35.47,133.05),
             '岡山':(34.66,133.93),'広島':(34.4,132.46),
             '山口':(34.19,131.47),'徳島':(34.07,134.56),
             '香川':(34.34,134.04),'愛媛':(33.84,132.77),
             '高知':(33.56,133.53),'福岡':(33.61,130.42),
             '佐賀':(33.25,130.3),'長崎':(32.74,129.87),
             '熊本':(32.79,130.74),'大分':(33.24,131.61),
             '宮崎':(31.91,131.42),'鹿児島':(31.56,130.56),
             '沖縄':(26.21,127.68)}

current_weather_url = 'http://api.openweathermap.org/data/2.5/weather'
forecast_url='http://api.openweathermap.org/data/2.5/forecast'
appid = 'c91707ce69fdcafd65b0900b7f9a0cc1'   #自分のIDを入れてください

def get_place(text):
    for pref in prefs:
        if pref in text:
            return pref
    return ""

def get_date(text):
    if "今日" in text:
        return "今日"
    elif "明日" in text:
        return "明日"
    else:
        return ""

def get_type(text):
    if "天気" in text:
        return "天気"
    elif "気温" in text:
        return "気温"
    else:
        return ""

def get_current_weather(lat,lon):
    response = requests.get("{}?lat={}&lon={}&lang=ja&units=metric&APPID={}".format(current_weather_url,lat,lon,appid))
    return response.json()

def get_tomorrow_weather(lat,lon):
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    tomorrow_noon = datetime.combine(tomorrow,time(12,0))
    timestamp = tomorrow_noon.timestamp()

    response = requests.get("{}?lat={}&lon={}&lang=ja&units=metric&APPID={}".format(forecast_url,lat,lon,appid))
    dic = response.json()

    for i in range(len(dic["list"])):
        dt = float(dic["list"][i]["dt"])
        if dt >= timestamp:
            return dic["list"][i]
    return ""


app = QtCore.QCoreApplication()
el = QtCore.QEventLoop()

sm = QtScxml.QScxmlStateMachine.fromFile('states.scxml')

sm.start()
el.processEvents()

print("SYS> こちらは天気情報案内システムです")

uttdic = {"ask_place":"地名を言ってください",
          "ask_date":"日付を言ってください",
          "ask_type":"情報種別を言ってください"}

current_state = sm.activeStateNames()[0]
print("current_state=",current_state)

sysutt = uttdic[current_state]
print("SYS>",sysutt)

while True:
    text = input("> ")
    if current_state == "ask_place":
        place = get_place(text)
        if place != "":
            sm.submitEvent("place")
            el.processEvents()
    elif current_state == "ask_date":
        date = get_date(text)
        if date != "":
            sm.submitEvent("date")
            el.processEvents()
    elif current_state == "ask_type":
        _type = get_type(text)
        if _type != "":
            sm.submitEvent("type")
            el.processEvents()

    current_state = sm.activeStateNames()[0]
    print("current_state=",current_state)

    if current_state == "tell_info":
        print("お伝えします")
        lat = latlondic[place][0]
        lon = latlondic[place][1]
        print(lat,lon)
        
        print("lat=",lat,"lon=",lon)
        if date == "今日":
            cw = get_current_weather(lat,lon)
            if _type == "天気":
                print("AAA",cw["weather"][0]["description"]+"です")
            elif _type == "気温":
                print(str(cw["main"]["temp"])+"度です")
        elif date == "明日":
            tw = get_tomorrow_weather(lat,lon)
            if _type == "天気":
                print(tw["weather"][0]["description"]+"です")
            elif _type == "気温":
                print(str(tw["main"]["temp"])+"度です")
        break
    else:
        sysutt = uttdic[current_state]
        print("SYS>",sysutt)


print("ご利用ありがとうございました")

#generate_da_samples.py
import re
import random
import json
import xml.etree.ElementTree

prefs = ['三重','京都','佐賀','兵庫','北海道','千葉','和歌山',
         '埼玉','大分','大阪','奈良','宮城','宮崎','富山','山口',
         '山形','山梨','岐阜','岡山','岩手','島根','広島','徳島',
         '愛媛','愛知','新潟','東京','栃木','沖縄','滋賀','熊本',
         '石川','神奈川','福井','福岡','福島','秋田','群馬','茨城',
         '長崎','長野','青森','静岡','香川','高知','鳥取','鹿児島']

dates = ["今日","明日"]

types = ["天気","気温"]

def random_generate(root):
    buf = ""
    if len(root) == 0:
        return root.text
    for elem in root:
        if elem.tag == "place":
            pref = random.choice(prefs)
            buf += pref
        elif elem.tag == "date":
            date = random.choice(dates)
            buf += date
        elif elem.tag == "type":
            _type =  random.choice(types)
            buf += _type
        if elem.tail is not None:
            buf += elem.tail
    return buf


fp = open("da_samples.dat","w")

da = ''

for line in open("examples.txt","r"):
    line = line.rstrip()
    if re.search(r'^da=',line):
        da = line.replace('da=','')
    elif line == "":
        pass
    else:
        root = xml.etree.ElementTree.fromstring("<dummy>"+line+"</dummy>")
        for i in range(1000):
            sample = random_generate(root)
            fp.write(da + "\t" + sample + "\n")


fp.close()

#train_da_model.py
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import dill

mecab = MeCab.Tagger()
mecab.parse('')

sents = []
labels = []

for line in open ("da_samples.dat","r"):
    line = line.rstrip()
    da,utt = line.split('\t')
    words = []
    for line in mecab.parse(utt).splitlines():
        if line == "EOS":
            break
        else:
            word,feature_str = line.split("\t")
            words.append(word)

    sents.append(" ".join(words))

    labels.append(da)


vectorizer = TfidfVectorizer(tokenizer=lambda x:x.split())
X = vectorizer.fit_transform(sents)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(labels)

svc = SVC(gamma="scale")
svc.fit(X,Y)

with open("svc.model","wb") as f:
    dill.dump(vectorizer,f)
    dill.dump(label_encoder,f)
    dill.dump(svc,f)
#da_extractor.py
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import dill

mecab = MeCab.Tagger()
mecab.parse('')

with open("svc.model","rb") as f:
    vectorizer = dill.load(f)
    label_encoder=dill.load(f)
    svc = dill.load(f)

def extract_da(utt):
    words=[]
    for line in mecab.parse(utt).splitlines():
        if line == "EOS":
            break
        else:
            word,feature_str = line.split("\t")
            words.append(word)

    tokens_str = " ".join(words)
    X = vectorizer.transform([tokens_str])
    Y = svc.predict(X)

    da = label_encoder.inverse_transform(Y)[0]
    return da
    
for utt in ["大阪の明日の天気","もう一度はじめから","東京じゃなくて"]:
    da = extract_da(utt)
    print(utt,da)
    
    #generate_concept_samples.py
import MeCab
import re
import random
import json
import xml.etree.ElementTree

# 都道府県名のリスト
prefs = ['三重', '京都', '佐賀', '兵庫', '北海道', '千葉', '和歌山', '埼玉', '大分',
         '大阪', '奈良', '宮城', '宮崎', '富山', '山口', '山形', '山梨', '岐阜', '岡山',
         '岩手', '島根', '広島', '徳島', '愛媛', '愛知', '新潟', '東京',
         '栃木', '沖縄', '滋賀', '熊本', '石川', '神奈川', '福井', '福岡', '福島', '秋田',
         '群馬', '茨城', '長崎', '長野', '青森', '静岡', '香川', '高知', '鳥取', '鹿児島']

# 日付のリスト
dates = ["今日","明日"]

# 情報種別のリスト
types = ["天気","気温"]

# サンプル文に含まれる単語を置き換えることで学習用事例を作成
def random_generate(root):
    buf = ""
    pos = 0
    posdic = {}
    # タグがない文章の場合は置き換えしないでそのまま返す
    if len(root) == 0:
        return root.text, posdic
    # タグで囲まれた箇所を同じ種類の単語で置き換える
    for elem in root:
        if elem.tag == "place":
            pref = random.choice(prefs)
            buf += pref
            posdic["place"] = (pos, pos+len(pref))
            pos += len(pref)
        elif elem.tag == "date":
            date = random.choice(dates)
            buf += date
            posdic["date"] = (pos, pos+len(date))
            pos += len(date)
        elif elem.tag == "type":
            _type =  random.choice(types)
            buf += _type
            posdic["type"] = (pos, pos+len(_type))
            pos += len(_type)
        if elem.tail is not None:
            buf += elem.tail
            pos += len(elem.tail)
    return buf, posdic

# 現在の文字位置に対応するタグをposdicから取得
def get_label(pos, posdic):
    for label, (start, end) in posdic.items():
        if start <= pos and pos < end:
            return label
    return "O"

# MeCabの初期化
mecab = MeCab.Tagger()
mecab.parse('')

# 学習用ファイルの書き出し先 
fp = open("concept_samples.dat","w")

da = ''
# eamples.txt ファイルの読み込み
for line in open("examples.txt","r"):
    line = line.rstrip()
    # da= から始まる行から対話行為タイプを取得
    if re.search(r'^da=',line):
        da = line.replace('da=','')
    # 空行は無視
    elif line == "":
        pass
    else:
        # タグの部分を取得するため，周囲にダミーのタグをつけて解析
        root = xml.etree.ElementTree.fromstring("<dummy>"+line+"</dummy>")
        # 各サンプル文を1000倍に増やす
        for i in range(1000):
            sample, posdic = random_generate(root)

            # lis は[単語，品詞，ラベル]のリスト
            lis = []
            pos = 0
            prev_label = ""
            for line in mecab.parse(sample).splitlines():
                if line == "EOS":
                    break
                else:
                    word, feature_str = line.split("\t")
                    features = feature_str.split(',')
                    # 形態素情報の0番目が品詞
                    postag = features[0]
                    # 現在の文字位置に対応するタグを取得
                    label = get_label(pos, posdic)
                    # label がOでなく，直前のラベルと同じであればラベルに'I-'をつける
                    if label == "O":
                        lis.append([word, postag, "O"])
                    elif label == prev_label:
                        lis.append([word, postag, "I-" + label])
                    else:
                        lis.append([word, postag, "B-" + label])
                    pos += len(word)
                    prev_label = label
            
            # 単語，品詞，ラベルを学習用ファイルに書き出す
            for word, postag, label in lis:
                fp.write(word + "\t" + postag + "\t" + label + "\n")
            fp.write("\n")

fp.close()


#train_concept_model.py
import json
import dill
import sklearn_crfsuite
from crf_util import word2features, sent2features, sent2labels

sents = []
lis = []

# concept_samples.dat の読み込み
for line in open("concept_samples.dat","r"):
    line = line.rstrip()
    # 空行で一つの事例が完了
    if line == "":
        sents.append(lis)
        lis = []
    else:
        # concept_samples.dat は単語，品詞，ラベルがタブ区切りになっている
        word, postag, label = line.split('\t')
        lis.append([word, postag, label])

# 各単語の情報を素性に変換
X = [sent2features(s) for s in sents]

# 各単語のラベル情報
Y = [sent2labels(s) for s in sents]

# CRFによる学習
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=False
)
crf.fit(X, Y)

# CRFモデルの保存
with open("crf.model","wb") as f:
    dill.dump(crf, f)
    
    
#concept_extractor.py
import MeCab
import json
import dill
import sklearn_crfsuite
from crf_util import word2features, sent2features, sent2labels
import re

# MeCabの初期化
mecab = MeCab.Tagger()
mecab.parse('')

# CRFモデルの読み込み
with open("crf.model","rb") as f:
    crf = dill.load(f)
    
# 発話文からコンセプトを抽出
def extract_concept(utt):
    lis = []
    for line in mecab.parse(utt).splitlines():
        if line == "EOS":
            break
        else:
            word, feature_str = line.split("\t")
            features = feature_str.split(',')
            postag = features[0]
            lis.append([word, postag, "O"])

    words = [x[0] for x in lis]            
    X = [sent2features(s) for s in [lis]]
    
    # 各単語に対応するラベル列
    labels = crf.predict(X)[0]
    
    # 単語列とラベル系列の対応を取って辞書に変換
    conceptdic = {}
    buf = ""
    last_label = ""
    for word, label in zip(words, labels):
        if re.search(r'^B-',label):
            if buf != "":
                _label = last_label.replace('B-','').replace('I-','')
                conceptdic[_label] = buf                    
            buf = word
        elif re.search(r'^I-',label):
            buf += word
        elif label == "O":
            if buf != "":
                _label = last_label.replace('B-','').replace('I-','')
                conceptdic[_label] = buf
                buf = ""
        last_label = label
    if buf != "":
        _label = last_label.replace('B-','').replace('I-','')
        conceptdic[_label] = buf
        
    return conceptdic

if __name__ ==  '__main__':
    for utt in ["大阪の明日の天気","もう一度はじめから","東京じゃなくて"]:    
        conceptdic = extract_concept(utt)
        print(utt, conceptdic)
        
        
#frame_weather2.py
from da_concept_extractor import DA_Concept

# 都道府県名のリスト
prefs = ['三重', '京都', '佐賀', '兵庫', '北海道', '千葉', '和歌山', '埼玉', '大分',
         '大阪', '奈良', '宮城', '宮崎', '富山', '山口', '山形', '山梨', '岐阜', '岡山',
         '岩手', '島根', '広島', '徳島', '愛媛', '愛知', '新潟', '東京',
         '栃木', '沖縄', '滋賀', '熊本', '石川', '神奈川', '福井', '福岡', '福島', '秋田',
         '群馬', '茨城', '長崎', '長野', '青森', '静岡', '香川', '高知', '鳥取', '鹿児島']

# 日付のリスト
dates = ["今日","明日"]

# 情報種別のリスト
types = ["天気","気温"]    

# システムの対話行為とシステム発話を紐づけた辞書
uttdic = {"open-prompt": "ご用件をどうぞ",
          "ask-place": "地名を言ってください",
          "ask-date": "日付を言ってください",
          "ask-type": "情報種別を言ってください"}

# 発話から得られた情報をもとにフレームを更新
def update_frame(frame, da, conceptdic):
    # 値の整合性を確認し，整合しないものは空文字にする
    for k,v in conceptdic.items():
        if k == "place" and v not in prefs:
            conceptdic[k] = ""
        elif k == "date" and v not in dates:
            conceptdic[k] = ""
        elif k == "type" and v not in types:
            conceptdic[k] = ""
    if da == "request-weather":
        for k,v in conceptdic.items():
            # コンセプトの情報でスロットを埋める
            frame[k] = v
    elif da == "initialize":
        frame = {"place": "", "date": "", "type": ""}
    elif da == "correct-info":
        for k,v in conceptdic.items():
            if frame[k] == v:
                frame[k] = ""
    return frame

# フレームの状態から次のシステム対話行為を決定
def next_system_da(frame):
    # すべてのスロットが空であればオープンな質問を行う
    if frame["place"] == "" and frame["date"] == "" and frame["type"] == "":
        return "open-prompt"
    # 空のスロットがあればその要素を質問する
    elif frame["place"] == "":
        return "ask-place"
    elif frame["date"] == "":
        return "ask-date"
    elif frame["type"] == "":
        return "ask-type"
    else:
        return "tell-info"

# 対話行為タイプとコンセプトの推定器
da_concept = DA_Concept()    

# フレーム
frame = {"place": "", "date": "", "type": ""}    

# システムプロンプト
print("SYS> こちらは天気情報案内システムです")
print("SYS> ご用件をどうぞ")

# ユーザ入力の処理
while True:
    text = input("> ")

    # 現在のフレームを表示
    print("frame=", frame)

    # 手入力で対話行為タイプとコンセプトを入力していた箇所を
    # 自動推定するように変更
    da, conceptdic = da_concept.process(text)        
    print(da, conceptdic)

    # 対話行為タイプとコンセプトを用いてフレームを更新
    frame = update_frame(frame, da, conceptdic)

    # 更新後のフレームを表示    
    print("updated frame=", frame)    

    # フレームからシステム対話行為を得る   
    sys_da = next_system_da(frame)

    # 遷移先がtell_infoの場合は情報を伝えて終了
    if sys_da == "tell-info":
        print("天気をお伝えします")
        break
    else:
        # 対話行為に紐づいたテンプレートを用いてシステム発話を生成
        sysutt = uttdic[sys_da]
        print("SYS>", sysutt)           

# 終了発話
print("ご利用ありがとうございました") 

