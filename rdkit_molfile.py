import cx_Oracle
import csv
from rdkit.Chem import MolFromMolBlock


USER = "ds3_userdata"
PASSWORD = "ds3_userdata"
DSN = cx_Oracle.makedsn(
    host="dotoradb-2022-0530-dev.fount.fount",
    port=1521,
    sid="orcl_dm",
)

pool = cx_Oracle.SessionPool(
    user=USER,
    password=PASSWORD,
    dsn=DSN,
    min=4,
    max=10,
    increment=1,
    encoding="UTF-8",
    wait_timeout=120,
    timeout=20,
)


def output_type_handler(cursor, name, default_type, size, precision, scale):
    if default_type == cx_Oracle.DB_TYPE_CLOB:
        return cursor.var(cx_Oracle.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if default_type == cx_Oracle.DB_TYPE_BLOB:
        return cursor.var(cx_Oracle.DB_TYPE_LONG_RAW, arraysize=cursor.arraysize)
    if default_type == cx_Oracle.DB_TYPE_NCLOB:
        return cursor.var(cx_Oracle.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)


def do_query(sql_stmt, bind_params=None):
    with pool.acquire() as conn:
        conn.outputtypehandler = output_type_handler
        cursor = conn.cursor()
        if bind_params:
            cursor.setinputsizes(VIR_ID=cx_Oracle.STRING, MOLFILE=cx_Oracle.CLOB)
            cursor.execute(sql_stmt, bind_params)
            conn.commit()
        else:
            cursor.execute(sql_stmt)
            rows = cursor.fetchall()
            return rows


def check_molstr(molfile_str):
    molecule = MolFromMolBlock(molfile_str, sanitize=True, strictParsing=True)
    if molecule:
        return 0
    return 1


def update_chiral_flag(molfile_str, vid, output_dct, cnt):
    try:
        molfile_lines = molfile_str.split("\n")
        counts_line = molfile_lines[3]
        chiral_flag = counts_line[12:15]

        output_dct["COUNT"].append(cnt)
        output_dct["VIR_ID"].append(vid)
        if chiral_flag.strip() == "1":
            counts_line = counts_line[:12] + " 0 " + counts_line[15:]

            molfile_lines[3] = counts_line

            updated_molfile_str = "\n".join(molfile_lines)
            print(f"{cnt}: CONVERTED {vid} -- {molfile_lines[3].strip()}")
            output_dct["CONVERTED"].append(1)
            output_dct["MOLFILE"].append(updated_molfile_str)
            output_dct["CHECK_INTEGRITY"].append(check_molstr(updated_molfile_str))

            return updated_molfile_str

        print(f"{cnt}: NOT {vid} -- {molfile_lines[3].strip()}")
        output_dct["CONVERTED"].append(0)
        output_dct["MOLFILE"].append(molfile_str)
        output_dct["CHECK_INTEGRITY"].append(check_molstr(molfile_str))
        return molfile_str
    except Exception as e:
        print(f"ERROR: update_chiral_flag - {vid} | {e}")


def write_dict_to_csv(data_dict, output_file):
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    num_rows = len(values[0])
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        for i in range(num_rows):
            row_data = [values[j][i] for j in range(len(keys))]
            writer.writerow(row_data)


if __name__ == "__main__":
    output_dct = {
        "COUNT": [],
        "CONVERTED": [],
        "VIR_ID": [],
        "MOLFILE": [],
        "CHECK_INTEGRITY": [],
    }
    mol_sql_stmt = "SELECT MOLFILE FROM ds3_userdata.FT_VIR_DESIGN WHERE VIR_ID = '{0}'"
    sql_stmt = """SELECT VIR_ID FROM
                ds3_userdata.FT_VIR_DESIGN
                WHERE MOLFILE IS NOT NULL
                AND VIR_ID IS NOT NULL"""
    vir_ids = do_query(sql_stmt)
    # print(vir_ids)
    for i, vid in enumerate(vir_ids):
        vid = vid[0]
        molfile = do_query(mol_sql_stmt.format(vid))
        try:
            # print(molfile[0][0])
            molfile_str = update_chiral_flag(molfile[0][0], vid, output_dct, i)
            insert_stmt = """INSERT INTO DS3_USERDATA.FT_VIR_CHIRAL_CLEAN
                            (VIR_ID, MOLFILE)
                            VALUES (:VIR_ID, :MOLFILE)"""
            params = {"VIR_ID": vid, "MOLFILE": molfile_str}
            do_query(insert_stmt, params)
        except Exception as e:
            print(f"ERROR: vir_id: {vid} - {e}")
    write_dict_to_csv(output_dct, "vir_id_molfile_chiral_flag.csv")
