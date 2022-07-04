import numpy as np
import psycopg2
import pandas.io.sql as psql

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from matplotlib import pyplot
import matplotlib.pyplot as plt


# from conn import CustomClass

# 센서ID리스트 호출 테스트
# CustomClass.queryData

def GetData(quary):
    try:
        conn = psycopg2.connect(host="3.38.51.236", dbname="mobiusdb", user="postgres", password="csahqldjtm", port="5432") # db에 접속
    except psycopg2.DatabaseError as db_err:
        print(db_err)
        return
    pythonTest = psql.read_sql(quary, conn)
    conn.close()
    return pythonTest

# 1시간 평균 균열
quary = "SELECT t, avg(\"value\") as val  "
quary += "FROM (SELECT time_bucket('1 days', time) as t, \"value\" FROM public.tb_static_data "
quary += "WHERE time >= '2021-10-01 00:00:00'   AND time < '2022-05-01 00:00:00'  AND channel_number = '2'"
quary += "AND device_id = 'Ssmartcs:2:DNAGW2111') AS ccc GROUP BY t ORDER BY t  " # 지정 마다 버킷에 넣는다.
sequence_length = 30
time_steps = 100
DAYS_TO_PREDICT = 50

# 신통교
pythonTest = GetData(quary)
print(pythonTest)
print("쿼리데이터 길이1 : " +str(len(pythonTest)))

##################### LSTM
# DBdata를 활용한 LSTM을 통한 예측
dataset = pythonTest
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
   
# plt.figure(1, figsize=(15,5))
# plt.plot(df["t"], df["val"], "-", color='black', label=str("spot_"))
# plt.legend()

print(rnn_real_predictions)
print("예측데이터 길이 : " +str(len(rnn_real_predictions)))
print(values)
print("쿼리데이터 길이2 : " +str(len(values)))

# pyplot.figure(3, figsize=(15,5))
# pyplot.plot(rnn_real_predictions, label='rnn_result')
# pyplot.plot(values, label='True_result')
# pyplot.legend()
# pyplot.show()