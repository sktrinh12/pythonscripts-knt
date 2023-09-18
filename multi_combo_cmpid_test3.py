import sys
import re

cmp_ids = [
    "FT002787",
    "FT009018",
    "FT008391",
    "FT007136",
    "FT007615",
    "FT009524",
    "FT004388",
]

number_of_cmpids = int(sys.argv[1])
# print(number_of_cmpids)

cmp_ids = cmp_ids[:number_of_cmpids]

CRO = "Pharmaron"
ASSAY_TYPE = "HTRF"
INC_HR = 1
PCT_SERUM = None

num_items = len(cmp_ids)

cte_template = """SELECT {cmpid_gmean_select},
    {tbl_1}.CRO,
    {tbl_1}.ASSAY_TYPE,
    {tbl_1}.CELL,
    {tbl_1}.VARIANT,
    {tbl_1}.INC_HR,
    {tbl_1}.PCT_SERUM
      FROM {tbl_1}
      LEFT OUTER JOIN {tbl_0}
      ON {tbl_1}.CRO = {tbl_0}.CRO
    AND {tbl_1}.CELL = {tbl_0}.CELL
    AND {tbl_1}.VARIANT = {tbl_0}.VARIANT
    AND {tbl_1}.ASSAY_TYPE = {tbl_0}.ASSAY_TYPE
    AND {tbl_1}.INC_HR = {tbl_0}.INC_HR
    AND {tbl_1}.PCT_SERUM = {tbl_0}.PCT_SERUM
    """

join_clause = ""
last_clause = ""
cte_clause = ""
cte_select_stmt = ""
cnt = 0

select_clause_lst = []
select_clause_enum_lst = []

# create individual with cte tables for each compound id
for i, cmp_id in enumerate(cmp_ids):
    if i < num_items:
        join_clause += f"""
            {f', t{i+1} AS ' if i>0 else f' t{i+1} AS' } (
            SELECT DISTINCT COMPOUND_ID, GEO_NM, CRO, ASSAY_TYPE, CELL, VARIANT, INC_HR, PCT_SERUM
            FROM SU_CELLULAR_DRC_STATS
            WHERE COMPOUND_ID = '{cmp_id}'
            AND CRO = '{ CRO }'
            AND ASSAY_TYPE = '{ ASSAY_TYPE }'
            AND INC_HR = { INC_HR }
            {'AND PCT_SERUM = ' + str(PCT_SERUM) if PCT_SERUM is not None else ''}
            )
            """
    # select column names w/o alias
    select_clause_lst.append(
        f"{{tbl_prefix}}{i+1}.COMPOUND_ID COMPOUND_ID_{i+1}, {{tbl_prefix}}{i+1}.GEO_NM GEO_NM_{i+1}"
    )
    # select column names /w alias
    select_clause_enum_lst.append(
        f"{{tbl_prefix}}{{nbr}}.COMPOUND_ID_{i+1}, {{tbl_prefix}}{{nbr}}.GEO_NM_{i+1}"
    )

# equivalent to ceil() without math library, add +1 since starting at 1
cte_loop_count = -int(-(num_items / 2) // 1) + 1

if num_items < 4:
    cte_loop_count -= 1
else:
    cte_loop_count += 1

# create cte inner subqueries
for i in range(1, cte_loop_count):
    cnt = i
    if i > 1:
        select_clause_tmp_lst = select_clause_enum_lst[: i + 1]
        select_clause_edit_lst = list(
            map(
                lambda x: x.format(tbl_prefix="cte_", nbr=i - 1),
                select_clause_tmp_lst[:i],
            )
        )
    else:
        select_clause_tmp_lst = select_clause_lst[: i + 1]
        select_clause_edit_lst = list(
            map(lambda x: x.format(tbl_prefix="t"), select_clause_tmp_lst[:i])
        )
    # append last one which should be a `t` prefixed table to list
    select_clause_edit_lst += list(
        map(lambda x: x.format(tbl_prefix="t", nbr=i + 1), select_clause_tmp_lst[i:])
    )
    # alias the last compound_id and geo_nm columns
    if i > 1:
        last_elem = select_clause_edit_lst[-1]
        cmp_el, gm_el = last_elem.split(",")
        new_cmp_el = cmp_el[:-2]
        new_cmp_el += f" COMPOUND_ID_{i+1}"
        new_gm_el = gm_el[:-2]
        new_gm_el += f" GEO_NM_{i+1}"
        new_last_elem = f"{new_cmp_el}, {new_gm_el}"
        select_clause_edit_lst[-1] = new_last_elem
        # print(new_last_elem)
    cte_select_stmt = ", ".join(select_clause_edit_lst)
    cte_clause += f""", cte_{i} as (
    {cte_template.format(tbl_1=f"{'cte_' if i > 1 else 't'}{i-1 if i > 1 else i}",
                         tbl_0=f"t{i+1}",
                         cmpid_gmean_select=cte_select_stmt)
    })
    """

select_clause_lst = []
for i in range(1, num_items + 1):
    select_clause = ""
    if i < num_items and num_items > 2:
        tbl_p = f"cte_{cnt}"
    else:
        tbl_p = f"t{i}"

    select_clause += (
        f" {tbl_p}.COMPOUND_ID{f'_{i}' if i < num_items and num_items>2 else ''}"
    )
    select_clause += (
        f", {tbl_p}.GEO_NM{f'_{i}' if i < num_items and num_items>2 else ''}"
    )
    select_clause_lst.append(select_clause)

# edit aliases for last select clause
select_clause_lst[-1] = re.sub(
    r"(COMPOUND_ID)", r"\1 \1_" + str(num_items), select_clause_lst[-1]
)
select_clause_lst[-1] = re.sub(
    r"(GEO_NM)", r"\1 \1_" + str(num_items), select_clause_lst[-1]
)
select_clause = ", ".join(select_clause_lst)

if num_items < 3:
    cnt += 1

end_clause = f""", cte_nested AS (
     {cte_template.format(tbl_1=f"{f'cte_{cnt}' if num_items>2 else f't{cnt}'}",
                         tbl_0=f"t{num_items}",
                         cmpid_gmean_select=select_clause)}
    )
    """

# strip the comma at the end if just one ft num and not include cte or end clause
sql_statement = f"""WITH {join_clause}
        {cte_clause}
        {end_clause if num_items > 1 else ''}
        SELECT * FROM {'cte_nested' if num_items > 1 else 't1'} ORDER BY CELL
        """

print(sql_statement)
