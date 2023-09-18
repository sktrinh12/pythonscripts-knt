import cx_Oracle
import datetime

oracle_dir = '/Users/spencer.trinhkinnate.com/instantclient_12_2/'

cx_Oracle.init_oracle_client(lib_dir=oracle_dir)

with open('/Users/spencer.trinhkinnate.com/Pictures/in_vivo_pk_data_img_upload_test.png', 'rb') as f:
    imgdata = f.read()


class OracleConnection(object):
    """Oracle DB Connection"""

    def __init__(self, username, password, hostname, port, sid):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.port = port
        self.sid = sid
        self.con = None
        self.dsn = cx_Oracle.makedsn(self.hostname, self.port, self.sid)

    def __enter__(self):
        try:
            self.con = cx_Oracle.connect(
                user=self.username, password=self.password, dsn=self.dsn)
            return self.con
        except cx_Oracle.DatabaseError:
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.con.close()
        except cx_Oracle.DatabaseError:
            pass


cred_dct = {}

cred_file = '/Users/spencer.trinhkinnate.com/Documents/security_files/oracle2'
with open(cred_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        str_split = line.split(',')
        key = str_split[0].strip()
        value = str_split[1].strip()
        cred_dct[key] = value

data = {"study_nbr": "PH-DMPK-KIN-22-076",
        "proj_id": "KIN-04",
        "request_by": "Ding Yuan",
        "study_dir": "Junyang Guo",
        "team_name": "PK-BA",
        "prot_type": "Rapid PK;Kp",
        "study_prot_id": "PK-R-RAPID_BP Kp-00",
        "date_revd": datetime.datetime.now(),
        "date_init": datetime.datetime.now(),
        "date_report": datetime.datetime.now(),
        "notes": "test notes",
        "image": imgdata,
        "audit_id": 1234,
        "expt_id": 987654,
        "doc_id": "DF5735E2F63629C0E053E902000A1234",
        "script_id": 2100,
        "img_base": None
        }

if __name__ == "__main__":
    with OracleConnection(cred_dct['USERNAME'],
                          cred_dct['PASSWORD'],
                          cred_dct['HOST-PROD'],
                          cred_dct['PORT'],
                          cred_dct['SID']) as con:
        with con.cursor() as cursor:
            cursor.execute("""INSERT INTO
                            DS3_USERDATA.COPY_FT_PHARM_STUDY
                            VALUES(:study_nbr, :proj_id, :request_by,
                                 :study_dir, :team_name, :prot_type,
                                 :study_prot_id, :date_revd, :date_init,
                                 :date_report, :notes, :image, :audit_id,
                                 :expt_id, :doc_id, :script_id, :img_base)""", data)
            con.commit()
