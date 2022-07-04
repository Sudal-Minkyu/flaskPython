from flask import Flask, request

# 권태호 박사 사용한 라이브러리
import psycopg2
import numpy as np
import pandas.io.sql as psql
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

app = Flask(__name__)

# http://192.168.0.143:8013/
@app.route("/")
def start():
    return "플라스크 테스트!"

# 센서리스트 데이터 불러오기
# http://192.168.0.143:8013/conn
@app.route("/conn", methods=['GET','POST'])
def sensorListData():
    print("센서리스트 데이터 불러오기 API 호출")
    try:
        conn = psycopg2.connect(
            host = '3.38.51.236',
            dbname = 'mobiusdb',
            user = 'postgres',
            password = 'csahqldjtm',
            port = '5432'
        )

        cur = conn.cursor()
        cur.execute("select a.device_id, a.device_name from public.tb_device a")
        rows = cur.fetchall()
        print()
        # print(rows)
        print("센서ID 총 길이 : "+str(len(rows)))
        print()
        return {
            'statusCode': 200,
            'body': rows
        }

    except:
        print("연결 실패!")
        return 0

# 센서 데이터 불러오기
# http://192.168.0.143:8013/sensorData?sensor=Ssmartcs:2:DNAGW2111&time1=2021-10-01%2000:00:00&time2=2022-05-01%2000:00:00
@app.route("/sensorData", methods=['GET','POST'])
def sensorData():
    print("센서 데이터 불러오기 API 호출")
    try:
        conn = psycopg2.connect(
            host = '3.38.51.236',
            dbname = 'mobiusdb',
            user = 'postgres',
            password = 'csahqldjtm',
            port = '5432'
        )

        cur = conn.cursor()

        sensor = request.args["sensor"] # 받은 조회 할 센서ID
        time1 = request.args["time1"] # 받은 시간 From
        time2 = request.args["time2"] # 받은 시간 To

        # sensor = 'Ssmartcs:2:DNAGW2111' # 받은 조회 할 센서ID
        # time1 = '2021-10-01 00:00:00' # 받은 시간 From
        # time2 = '2022-05-01 00:00:00' # 받은 시간 To

        quary = f"SELECT t, avg(\"value\") as val "
        quary += f"FROM (SELECT time_bucket('1 days', time) as t, \"value\" FROM public.tb_static_data "
        quary += f"WHERE time >= '{time1}' AND time < '{time2}' AND channel_number = '2' "
        quary += f"AND device_id = '{sensor}') AS ccc GROUP BY t ORDER BY t".format(time1=time1,time2=time2,sensor=sensor)
        # sequence_length = 30
        # time_steps = 100
        # DAYS_TO_PREDICT = 50

        cur.execute(quary)

        rows = cur.fetchall()

        print("쿼리데이터 길이 : " +str(len(rows)))

        return {
                'statusCode': 200,
                'body': rows
            }
    
    except psycopg2.DatabaseError as db_err:
        print(db_err)
        return

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8013)