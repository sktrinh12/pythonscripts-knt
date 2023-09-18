import cx_Oracle
import psycopg2
import time

oracle_username = "ds3_userdata"
oracle_password = "ds3_userdata"
oracle_dsn = cx_Oracle.makedsn(
    host="dotoradb-2022-0530-dev.fount.fount",
    port=1521,
    sid="orcl_dm",
)

pg_host = "postgres.kinnate"
pg_dbname = "postgres"
pg_username = "postgres"
pg_password = "kinnate"


def fetch_data_from_oracle():
    oracle_connection = cx_Oracle.connect(oracle_username, oracle_password, oracle_dsn)
    oracle_cursor = oracle_connection.cursor()

    oracle_cursor.execute(
        """SELECT COMPOUND_ID, CRO, PROJECT, ASSAY_TYPE, TARGET,
                          VARIANT, COFACTORS, THIOL_FREE, ATP_CONC_UM, GEO_NM,
                          N_OF_M, CREATED_DATE FROM
                          SU_BIOCHEM_DRC_STATS""",
    )
    data = oracle_cursor.fetchall()

    oracle_cursor.close()
    oracle_connection.close()

    return data


def create_staging_table(pg_cursor):
    pg_cursor.execute(
        """
        DROP TABLE IF EXISTS su_biochem_drc_stats_staging
    """
    )
    print("Staging table dropped")

    pg_cursor.execute(
        """CREATE TABLE su_biochem_drc_stats_staging (
    compound_id VARCHAR(32),
    cro VARCHAR(2000),
    project VARCHAR(2000),
    assay_type VARCHAR(1500),
    target VARCHAR(1500),
    variant VARCHAR(1500),
    cofactors VARCHAR(1500),
    thiol_free VARCHAR(100),
    atp_conc_um VARCHAR(100),
    geo_nm NUMERIC,
    n_of_m VARCHAR(100),
    created_date TIMESTAMP
    )"""
    )
    print("Created staging table")


def insert_data_into_postgres(pg_cursor, data):
    start_time = time.time()
    for row in data:
        print(row)
        pg_cursor.execute(
            """insert into su_biochem_drc_stats_staging (compound_id, cro, project,
                                               assay_type, target, variant,
                                               cofactors, thiol_free,
                                               atp_conc_um, geo_nm, n_of_m,
                                               created_date) values (%s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            row,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")


def drop_table(pg_cursor):
    pg_cursor.execute(
        """
        DROP TABLE IF EXISTS su_biochem_drc_stats
    """
    )
    print("Drop old table")


def alter_table(pg_cursor):
    pg_cursor.execute(
        """
        ALTER TABLE su_biochem_drc_stats_staging RENAME TO su_biochem_drc_stats
    """
    )
    print("Renamed staging table")


if __name__ == "__main__":
    try:
        data_to_sync = fetch_data_from_oracle()
        pg_connection = psycopg2.connect(
            host=pg_host, dbname=pg_dbname, user=pg_username, password=pg_password
        )
        pg_connection.autocommit = False
        pg_cursor = pg_connection.cursor()

        create_staging_table(pg_cursor)
        pg_connection.commit()

        insert_data_into_postgres(pg_cursor, data_to_sync)

        try:
            drop_table(pg_cursor)
            alter_table(pg_cursor)
            pg_connection.commit()
        except Exception as e:
            pg_connection.rollback()
            print("Error occurred during table drop and alter:", e)

        pg_cursor.close()
        pg_connection.close()
        print("Data synchronization successful for SU_BIOCHEM_DRC_STATS")
    except Exception as e:
        print(f"Error occurred during data synchronization: {e}")
