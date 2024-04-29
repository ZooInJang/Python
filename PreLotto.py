import schedule
import time
import os
import datetime
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ast
import warnings

# 불필요한 로그 삭제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL messages only
os.environ['PYTHONWARNINGS'] = 'ignore'   # Ignore Python warnings
warnings.filterwarnings('ignore')


# 스크래핑 모듈
def scrape():

    #스크래핑할 홈페이지
    url="https://www.dhlottery.co.kr/gameResult.do?method=byWin" #로또 당첨번호 홈페이지 주소

    #url을 읽어오고 내용 가져오기
    request = requests.get(url)
    data = request.text

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(data, 'html.parser')

    # 'div' 태그와 클래스 'num win'을 가진 모든 요소를 찾습니다.
    data_tags = soup.find_all('div', {'class': 'nums'})

    #추출해서 저장할 배열 변수
    numbers=[]
    
    # 각 'div' 태그에 대해
    for data_tag in data_tags:
        # 'span' 태그를 찾습니다.
        number_tags = data_tag.find_all('span')
        
        # 각 'span' 태그의 텍스트를 추출합니다.
        numbers = [tag.text for tag in number_tags]
        
        # 번호를 출력합니다.
        print(numbers)

    return numbers

#스크래핑한 값을 지정한 파일에 저장하는 작업의 모듈
def job():
    numbers = scrape()  # 스크래핑된 데이터
    last_line = None
    if os.path.exists('data.txt'):
        with open('data.txt', 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()  # 파일의 마지막 줄을 읽습니다.
    # 마지막 줄의 데이터와 스크래핑된 데이터를 비교합니다.
    if last_line != str(numbers):
        with open('data.txt', 'a') as f:
            f.write("%s\n" % numbers)  # 가장 최근의 데이터만 추가
    with open('last_run.json', 'w') as f:
        json.dump({'last_run': str(datetime.datetime.now())}, f)

# 프로그램 시작 시 마지막 실행 시간 확인
if os.path.exists('last_run.json'):
    with open('last_run.json', 'r') as f:
        last_run = json.load(f)['last_run']
        last_run = datetime.datetime.strptime(last_run, '%Y-%m-%d %H:%M:%S.%f')
        if datetime.datetime.now() - last_run > datetime.timedelta(days=7):
            job()

#프로그램 시작 전에 먼저 1번 실행
job()

# 예측 함수
def predict():
    # txt 파일에서 데이터 읽기
    with open('data.txt', 'r') as f:
        lines = f.readlines()

    # 가장 최근의 20줄만 사용
    lines = lines[-20:]

    # 데이터를 숫자로 변환
    #data = [list(map(int, line.strip().replace("'","").split())) for line in lines]
    data = [ast.literal_eval(line.strip()) for line in lines]

    # 데이터 정규화 (0~1 사이의 값으로 변환)
    normalized_data = np.array(data,dtype=float) / 50.0

    # LSTM에 입력하기 위해 차원 변경
    normalized_data = np.reshape(normalized_data, (normalized_data.shape[0], normalized_data.shape[1], 1))

    # 모델 구성
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(7, 1)))
    model.add(Dense(7))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    model.fit(normalized_data, normalized_data, epochs=200, verbose=0)

    # 다음에 올 7개의 숫자 예측
    test_input = np.array([np.array(data[-1],dtype=float) / 45.0])  # 가장 최근의 데이터를 사용
    test_input = np.reshape(test_input, (test_input.shape[0], test_input.shape[1], 1))
    prediction = model.predict(test_input)

    # 예측 결과를 원래의 범위(1~45)로 변환
    prediction = prediction * 45

    # 예측 결과를 정수로 변환
    prediction1 = np.round(prediction).astype(int)
    prediction2 = np.floor(prediction).astype(int)
    prediction3 = np.ceil(prediction).astype(int)

    print("반올림한 값 : ",*prediction1[0])#반올림
    print("내림 값 : ",*prediction2[0])#내림
    print("올림 값 : ",*prediction3[0])#올림


# 매주 금요일 저녁 7시에 웹 스크래핑 및 데이터 추가
#schedule.every().saturday.at("21:00").do(scrape)
schedule.every().seconds.do(job)

# 메뉴
while True:
    print("1: 예측 실행")
    print("2: 종료")
    choice = input("원하는 작업을 선택하세요: ")

    if choice == '1':
        # 예측 실행
        predict()
    elif choice == '2':
        # 종료
        break
    else:
        print("잘못된 선택입니다. 다시 선택해주세요.")

    # 스케줄된 작업이 있으면 실행
    schedule.run_pending()
    time.sleep(1)