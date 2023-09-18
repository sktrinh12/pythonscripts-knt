cmp_ids = ["FT002787", "FT009018", "FT004381"]
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

join_clause = ""
for i, cmp_id in enumerate(cmp_ids):
    join_clause += f"""
        {'LEFT OUTER JOIN' if i>0 else '' } (
        SELECT DISTINCT COMPOUND_ID, GEO_NM, CRO, ASSAY_TYPE, CELL, VARIANT, INC_HR, PCT_SERUM
        FROM SU_CELLULAR_DRC_STATS
        WHERE COMPOUND_ID = '{cmp_id}' AND CRO = '{ CRO }' AND ASSAY_TYPE = '{ ASSAY_TYPE }' {'AND PCT_SERUM = ' + str(PCT_SERUM) if PCT_SERUM is not None else ''}
        ) t{i+1}
        """
    if i > 0:
        join_clause += f"""
        ON t{i+1}.CRO = t{i}.CRO
        AND t{i+1}.CELL = t{i}.CELL
        AND t{i+1}.VARIANT = t{i}.VARIANT
        AND t{i+1}.ASSAY_TYPE = t{i}.ASSAY_TYPE
        AND t{i+1}.INC_HR = t{i}.INC_HR
        AND t{i+1}.PCT_SERUM = t{i}.PCT_SERUM
        """

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
                ORDER BY  {', '.join([f'COMPOUND_ID_{j}' for j in range(1, len(cmp_ids)+1)]) + ',' if len(cmp_ids) > 1 else ''} CELL, VARIANT
                """

print(sql_statement)
