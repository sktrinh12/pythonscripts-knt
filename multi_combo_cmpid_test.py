cmp_ids = ["FT002787", "FT009018"]
CRO = "Pharmaron"
ASSAY_TYPE = "HTRF"
INC_HR = 1
PCT_SERUM = 10

select_columns = [
    "t1.CRO",
    "t1.ASSAY_TYPE",
    "t1.CELL",
    "t1.VARIANT",
    "t1.INC_HR",
    "t1.PCT_SERUM",
]

where_conditions = []

for i, cmp_id in enumerate(cmp_ids):
    where_conditions.append(f"t{i+1}.COMPOUND_ID = '{cmp_id}'")

where_conditions.append(
    f"""
        t1.CRO = '{ CRO }'
        AND t1.ASSAY_TYPE = '{ ASSAY_TYPE }'
        AND t1.INC_HR = { INC_HR }
        AND t1.PCT_SERUM = { PCT_SERUM }
        """
)
where_clause = " AND ".join(where_conditions)

join_clause = ""
for i, cmp_id in enumerate(cmp_ids):
    if i > 0:
        join_clause += f""" INNER JOIN SU_CELLULAR_DRC_STATS t{i+1} ON
            t{i+1}.CRO = t{i}.CRO
            AND t{i+1}.CELL = t{i}.CELL
            AND t{i+1}.ASSAY_TYPE = t{i}.ASSAY_TYPE
            AND t{i+1}.INC_HR = t{i}.INC_HR
            AND t{i+1}.PCT_SERUM = t{i}.PCT_SERUM
            """
    else:
        join_clause += f" SU_CELLULAR_DRC_STATS t{i+1}"

select_clause = ", ".join(
    [
        f"t{i+1}.COMPOUND_ID COMPOUND_ID_{i+1}, t{i+1}.GEO_NM GEO_NM_{i+1}"
        if i > 0
        else f"t{i+1}.COMPOUND_ID COMPOUND_ID_1, t{i+1}.GEO_NM GEO_NM_1"
        for i in range(len(cmp_ids))
    ]
)
select_clause += ", " + ", ".join(select_columns)

sql_statement = f"""SELECT {select_clause}
                FROM {join_clause}
                WHERE {where_clause}
                ORDER BY CELL, VARIANT
                """

print(sql_statement)
