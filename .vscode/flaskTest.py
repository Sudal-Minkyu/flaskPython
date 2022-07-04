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

        #cur = conn.cursor()

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

        # cur.execute(quary)

        # rows = cur.fetchall()

        rows = psql.read_sql(quary, conn)

        # print("쿼리데이터 길이 : " +str(len(rows)))

        print("LSTM 호출시작")

        sequence_length = 30
        time_steps = 100
        DAYS_TO_PREDICT = 50

        # LSTM
        # DBdata를 활용한 LSTM을 통한 예측
        dataset = rows
        values = dataset.values
        values = values[:, 1]

        # ensure all data is float
        values = values.astype('float32')
        values = values.reshape([values.shape[0], 1])
        sc = MinMaxScaler(feature_range=(0, 1))
        scaled = sc.fit_transform(values)

        # split into train and test sets
        train = scaled
        ts_train_len = len(train)
        X_train_sq = []

        for i in range(sequence_length, ts_train_len):
            X_train_sq.append(train[i - sequence_length:i, 0])
        X_train_sq = np.array(X_train_sq)
        X_train = []
        X_train_sq_i = []

        for i in range(0, X_train_sq.shape[0]-time_steps+1):
            X_train_sq_i = X_train_sq[i:i+time_steps, :]
            X_train.append(X_train_sq_i)
        
        X_train = np.array(X_train)
        print(X_train.shape)

        model = Sequential()
        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(50, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        model.load_weights('testmodel.h5')

        predict_seq = X_train[-1:, :, :]
        predict_seq = np.delete(predict_seq, 0, axis=1)
        predict_seq_a = predict_seq[0, -1, :]
        predict_seq_a = np.delete(predict_seq_a, 0, axis=0)
        predict_seq_last = scaled[-1]
        predict_seq_a = np.concatenate((predict_seq_a, predict_seq_last), axis=0)
        predict_seq_a = np.reshape(predict_seq_a, (1, 1, predict_seq_a.shape[0]))
        predict_seq = np.concatenate((predict_seq, predict_seq_a), axis=1)
        rnn_real_predictions = values
        for _ in range(DAYS_TO_PREDICT):
            y_test_pred_sc = model.predict(predict_seq)
            y_test_pred = sc.inverse_transform(y_test_pred_sc)
            rnn_real_predictions = np.vstack([rnn_real_predictions, y_test_pred])
    
       
        print("쿼리데이터 길이1 : " +str(len(values)))
        print("예측데이터 길이2 : " +str(len(rnn_real_predictions)))
        # print(values)
        # print(rnn_real_predictions)

        # print(type(rnn_real_predictions))
        # print(type(values))

        queryList2 = rnn_real_predictions.tolist()
        queryList1 = values.tolist()

        return {
            'statusCode': 200,
            'queryList1': queryList1,
            'queryList1_Length': len(queryList1),
            'queryList2': queryList2,
            'queryList2_Length': len(queryList2)
        }

        # return {
        #         'statusCode': 200,
        #         'body': rows
        #     }
    
    except psycopg2.DatabaseError as db_err:
        print(db_err)
        print("LSTM 호출실패")
        return

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8013)