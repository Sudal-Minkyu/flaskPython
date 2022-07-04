
import psycopg2

class CustomClass:
    def queryData():
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
            print(rows)
            print("데이터 총 길이 : "+str(len(rows)))
            print()
            return {
                'statusCode': 200,
                'body': rows
            }

        except:
            print("연결 실패!")
            return 0

    queryData()