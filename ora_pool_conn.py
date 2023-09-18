import cx_Oracle
import threading
import time
import queue
import json


USER = "ds3_userdata"
PASSWORD = "ds3_userdata"
DSN = cx_Oracle.makedsn(
    host="dotoradb-2022-dev.fount",
    port=1521,
    sid="orcl_dm",
)

pool = cx_Oracle.SessionPool(
    user=USER,
    password=PASSWORD,
    dsn=DSN,
    min=6,
    max=10,
    increment=1,
    threaded=True,
    encoding="UTF-8",
    wait_timeout=120,
    timeout=20,
)


def get_sql_stmts():
    return [
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT002787' fetch next 2 rows only",
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT007615' fetch next 2 rows only",
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT008891' fetch next 2 rows only",
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT000953' fetch next 2 rows only",
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT004324' fetch next 2 rows only",
        "select COMPOUND_ID, IC50_NM, CELL_LINE from ds3_userdata.su_cellular_growth_drc where compound_id = 'FT004400' fetch next 2 rows only",
    ]


def do_query(sql_stmt, queue):
    with pool.acquire() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_stmt)
        rows = cursor.fetchall()
        queue.put(rows)
        # print(rows[0])


queue = queue.Queue()
start_time = time.time()

threads = [threading.Thread(target=do_query, args=(p, queue)) for p in get_sql_stmts()]
for t in threads:
    t.start()
for t in threads:
    t.join()

# for sql in get_sql_stmts():
#     do_query(sql)

end_time = time.time()

print("Elapsed time:", end_time - start_time)

results = []
while not queue.empty():
    results.append(queue.get())

json_payload = json.dumps(results)
print(json_payload)
