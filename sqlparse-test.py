import sqlparse
import re

subquery_exp = [
    "cro",
    "assay_type",
    "compound_id",
    "batch_id",
    "target",
    "variant",
    "cofactors",
    "atp_conc_um",
    "modifier",
    "geomean_nm",
    "n",
    "m",
]

subquery_cell_geomean = [
    "cro",
    "assay_type",
    "compound_id",
    "batch_id",
    "cell_line",
    "variant",
    "cell_incubation_hr",
    "pct_serum",
    "threed",
    "treatment",
    "geomean_nm",
    "created_date",
    "modifier",
    "flag",
    "n",
    "m",
]

main_query_exp = [
    "cro",
    "assay_type",
    "target",
    "variant",
    "cofactors",
    "geo_nM",
    "n_of_m",
]


def extract_exps(column_identifiers, expressions):
    select_lines = []
    expressions = [
        re.compile(r"\b" + re.escape(exp) + r"\b", re.IGNORECASE) for exp in expressions
    ]
    for column_identifier in column_identifiers:
        column_name = column_identifier.get_name()
        # print(column_name)
        for exp in expressions:
            if exp.search(str(column_name)):
                select_lines.append(str(column_identifier))
                break
    return select_lines


def extract_select_expr(parsed_stmt, expressions):
    column_identifiers = []

    in_select = False
    for token in parsed_stmt.tokens:
        if isinstance(token, sqlparse.sql.Comment):
            continue
        if str(token).lower() == "select":
            in_select = True
        elif in_select and token.ttype is None:
            for identifier in token.get_identifiers():
                column_identifiers.append(identifier)
                print(identifier)
            break
    select_lines = extract_exps(column_identifiers, expressions)
    return select_lines


def extract_from_clause(parsed_stmt):
    from_clause = "FROM "
    in_from = False

    for token in parsed_stmt.tokens:
        if token.match(sqlparse.tokens.Keyword, "FROM"):
            in_from = True
        elif in_from and not token.is_whitespace:
            if token.match(sqlparse.tokens.Keyword, "WHERE"):
                break
            from_clause += str(token)

    from_clause = re.sub(r"ds3", r" ds3", from_clause, flags=re.IGNORECASE)
    from_clause = re.sub(r"LEFT", " LEFT", from_clause, flags=re.IGNORECASE)
    from_clause = re.sub(r"WHERE", "  WHERE", from_clause, flags=re.IGNORECASE)
    from_clause = re.sub(
        r"([t|T]\d+)(ON)([t|T]\d+)", r"\1 \2 \3", from_clause, flags=re.IGNORECASE
    )
    return from_clause


def extract_subquery(parsed_stmt, select_exprs):
    subquery_section = []
    subquery_select_lines = []
    subquery_parsed_stmt = ""
    in_subquery = False

    for token in parsed_stmt.tokens:
        if isinstance(token, sqlparse.sql.Comment):
            continue
        if str(token).strip().lower().startswith("from"):
            in_subquery = True
        elif in_subquery and isinstance(token, sqlparse.sql.TokenList):
            str_token = str(token).strip()
            subquery_section.append(str_token)
            if str_token.endswith(") t0"):
                in_subquery = False

    subquery_string = subquery_section[0].strip("(").replace(" t0", "").rstrip(")")
    # print(subquery_string)
    subquery_parsed_stmt = sqlparse.parse(subquery_string)[0]
    subquery_select_lines = extract_select_expr(subquery_parsed_stmt, select_exprs)
    # subquery_where_clause = str(subquery_parsed_stmt[-1])
    subquery_from_clause = extract_from_clause(subquery_parsed_stmt)

    return subquery_select_lines, subquery_from_clause


def extract_where_clause(parsed_stmt, nesting_level=0):
    where_clause = ""

    for token in parsed_stmt.tokens:
        if isinstance(token, sqlparse.sql.Where) and nesting_level == 0:
            where_clause += str(token)
        elif isinstance(token, sqlparse.sql.TokenList):
            where_clause += extract_where_clause(token, nesting_level + 1)

    return where_clause.strip()


sql_statement = """ SELECT
    MAX(t0.cro)              AS cro,
    MAX(t0.assay_type)       AS assay_type,
    MAX(t0.compound_id)      AS compound_id,
    MAX(t0.target)           AS target,
    MAX(t0.variant)          AS variant,
    MAX(t0.cofactors)        AS cofactors,
    MAX(t0.atp_conc_um)      AS atp_conc_um,
    MAX(t0.geomean_nm)       AS geo_nm,
    MAX(t0.nm_minus_3_stdev) AS nm_minus_3_stdev,
    MAX(t0.nm_plus_3_stdev)  AS nm_plus_3_stdev,
    MAX(t0.nm_minus_3_var)   AS nm_minus_3_var,
    MAX(t0.nm_plus_3_var)    AS nm_plus_3_var,
    MAX(t0.n)
    || '
of
'
    || MAX(t0.m)             AS n_of_m,
    MAX(t0.cellvalue_two)    AS cellvalue_two
FROM
    (
        SELECT
            cro,
            assay_type,
            compound_id,
            batch_id,
            target,
            variant,
            cofactors,
            atp_conc_um,
            modifier,
            round(power(10, AVG(log(10, ic50))
                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,
                                              cofactors, atp_conc_um, modifier)) * to_number('1.0e+09'), 1)  AS geomean_nm,
            round(power(10, AVG(log(10, ic50))
                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,
                                              cofactors, atp_conc_um, modifier) -(3 * STDDEV(log(10, t1.ic50))
                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,
                                                                                      target, variant,
                                                                                                        cofactors, atp_conc_um, modifier))) *
                                                                                                        to_number('1.0e+09'), 1) AS nm_minus_3_stdev,
            round(power(10, AVG(log(10, ic50))
                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,
                                              cofactors, atp_conc_um, modifier) +(3 * STDDEV(log(10, t1.ic50))
                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,
                                                                                      target, variant,
                                                                                                        cofactors, atp_conc_um, modifier))) *
                                                                                                        to_number('1.0e+09'), 1) AS nm_plus_3_stdev,
            round(power(10, AVG(log(10, ic50))
                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,
                                              cofactors, atp_conc_um, modifier) -(3 * VARIANCE(log(10, t1.ic50))
                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,
                                                                                      target, variant,
                                                                                                        cofactors, atp_conc_um, modifier))) *
                                                                                                        to_number('1.0e+09'), 1) AS nm_minus_3_var,
            round(power(10, AVG(log(10, ic50))
                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,
                                              cofactors, atp_conc_um, modifier) +(3 * VARIANCE(log(10, t1.ic50))
                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,
                                                                                      target, variant,
                                                                                                        cofactors, atp_conc_um, modifier))) *
                                                                                                        to_number('1.0e+09'), 1) AS nm_plus_3_var,
            COUNT(t1.ic50)
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,
                              t1.cofactors, t1.atp_conc_um, t1.modifier)               AS n,
            COUNT(t1.ic50)
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,
                              t1.cofactors, t1.atp_conc_um)                            AS m,
            t1.cro
            || '|'
            || t1.assay_type
            || '|'
            || t1.target
            || '|'
            || nvl(t1.variant, '-')
            || '|'
            || nvl(t1.atp_conc_um, '-')
            || '|'
            || nvl(t1.cofactors, '-')                                      AS cellvalue_two
        FROM
            ds3_userdata.su_biochem_drc t1
        WHERE
                assay_intent = 'Screening'
            AND validated = 'VALIDATED'
    ) t0
WHERE
    t0.modifier IS NULL
GROUP BY
    t0.compound_id,
    t0.cro,
    t0.assay_type,
    t0.target,
    t0.variant,
    t0.cofactors,
    t0.atp_conc_um
"""

# parsed_stmt = sqlparse.parse(sql_statement)[0]
# main_select_lines = extract_select_expr(parsed_stmt, main_query_exp)
# subquery_select_lines, subquery_where_clause, subquery_from_clause = extract_subquery(
#     parsed_stmt
# )

# group_by_expression = extract_group_by(parsed_stmt)

# where_clause = extract_where_clause(parsed_stmt)

# test_sql_str = [
#     "SELECT            cro,            assay_type,            compound_id,            batch_id,            target,            variant,            cofactors,            atp_conc_um,            modifier,            round(power(10, AVG(log(10, ic50))                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,                                              cofactors, atp_conc_um, modifier)) * to_number('1.0e+09'), 1)  AS geomean_nm,            round(power(10, AVG(log(10, ic50))                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,                                              cofactors, atp_conc_um, modifier) -(3 * STDDEV(log(10, t1.ic50))                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,                                                                                      target, variant,                                                                                                        cofactors, atp_conc_um, modifier))) *                                                                                                        to_number('1.0e+09'), 1) AS nm_minus_3_stdev,            round(power(10, AVG(log(10, ic50))                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,                                              cofactors, atp_conc_um, modifier) +(3 * STDDEV(log(10, t1.ic50))                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,                                                                                      target, variant,                                                                                                        cofactors, atp_conc_um, modifier))) *                                                                                                        to_number('1.0e+09'), 1) AS nm_plus_3_stdev,            round(power(10, AVG(log(10, ic50))                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,                                              cofactors, atp_conc_um, modifier) -(3 * VARIANCE(log(10, t1.ic50))                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,                                                                                      target, variant,                                                                                                        cofactors, atp_conc_um, modifier))) *                                                                                                        to_number('1.0e+09'), 1) AS nm_minus_3_var,            round(power(10, AVG(log(10, ic50))                            OVER(PARTITION BY cro, assay_type, compound_id, target, variant,                                              cofactors, atp_conc_um, modifier) +(3 * VARIANCE(log(10, t1.ic50))                                                                                      OVER(PARTITION BY cro, assay_type, compound_id,                                                                                      target, variant,                                                                                                        cofactors, atp_conc_um, modifier))) *                                                                                                        to_number('1.0e+09'), 1) AS nm_plus_3_var,            COUNT(t1.ic50)            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,                              t1.cofactors, t1.atp_conc_um, t1.modifier)               AS n,            COUNT(t1.ic50)            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,                              t1.cofactors, t1.atp_conc_um)                            AS m,            t1.cro            || '|'            || t1.assay_type            || '|'            || t1.target            || '|'            || nvl(t1.variant, '-')            || '|'            || nvl(t1.atp_conc_um, '-')            || '|'            || nvl(t1.cofactors, '-')                                      AS cellvalue_two        FROM            ds3_userdata.su_biochem_drc t1        WHERE                assay_intent = 'Screening'            AND validated = 'VALIDATED' "
# ]
test_sql_str = """SELECT\r\n    max(t0.cro) AS CRO,\r\n    max(t0.project) AS project,\r\n    max(t0.assay_type) AS assay_type,\r\n    max(t0.compound_id) AS compound_id,\r\n    max(t0.cell_line) AS cell,\r\n    max(t0.variant) AS variant,\r\n    max(t0.cell_incubation_hr) AS inc_hr,\r\n    max(t0.pct_serum) AS pct_serum,\r\n    max(t0.threed) AS threed,\r\n    max(t0.treatment) AS treatment,\r\n    max(t0.geomean_nM) AS geo_nM,\r\n    max(t0.nm_minus_3_stdev) AS nm_minus_3_stdev,\r\n    max(t0.nm_plus_3_stdev) AS nm_plus_3_stdev,\r\n    max(t0.nm_minus_3_var) AS nm_minus_3_var,\r\n    max(t0.nm_plus_3_var) AS nm_plus_3_var,\r\n    max(t0.n) || ' of ' || max(t0.m) AS n_of_m,\r\n    max(t0.cellvalue_two) as cellvalue_two,\r\n    max(t0.stdev) as stdev,\r\n    max(t0.created_date) as created_date \r\nFROM (\r\n    SELECT\r\n        t1.cro,\r\n        t1.project,\r\n        t1.assay_type,\r\n        t1.compound_id,\r\n        t1.batch_id,\r\n        t1.cell_line,\r\n        nvl(t1.variant, '-') AS variant,\r\n        t1.created_date, \r\n        t1.cell_incubation_hr,\r\n        t1.pct_serum,\r\n        t2.flag,\r\n        case t1.threed when 'Y' then '3D' else '-' end AS threed,\r\n        case when t1.treatment is not null then treatment else '-' end AS treatment,\r\n        t1.modifier,\r\n        round(stddev(t1.ic50) OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier) * 1000000000, 2) AS stdev,\r\n        round((to_char(power(10, avg(log(10, t1.ic50)) OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier)), '99999.99EEEE') * 1000000000), 1) AS geomean_nM,\r\n        round(ABS(power(10, AVG(log(10, t1.ic50))\r\n                      OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant,\r\n                                        t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier))* 1000000000 \r\n                                        - (3 * STDDEV(t1.ic50)\r\n                                                                                                OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.\r\n                                                                                                cell_line, t1.variant, t1.cell_incubation_hr,\r\n                                                                                                t1.pct_serum,t1.threed, t1.treatment, t1.modifier) * 1000000000)), 3) AS nm_minus_3_stdev,\r\n        round(ABS(power(10, AVG(log(10, t1.ic50))\r\n                      OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant,\r\n                                        t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier))* 1000000000 \r\n                                        + (3 * STDDEV(t1.ic50)\r\n                                                                                                OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.\r\n                                                                                                cell_line, t1.variant, t1.cell_incubation_hr,\r\n                                                                                                t1.pct_serum,t1.threed, t1.treatment, t1.modifier) * 1000000000)), 3) AS nM_plus_3_stdev,\r\n        round(ABS(power(10, AVG(log(10, t1.ic50))\r\n                      OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant,\r\n                                        t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier))* 1000000000 \r\n                                        - (3 * VARIANCE(t1.ic50)\r\n                                                                                                OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.\r\n                                                                                                cell_line, t1.variant, t1.cell_incubation_hr,\r\n                                                                                                t1.pct_serum,t1.threed, t1.treatment, t1.modifier) * 1000000000)), 3) AS nm_minus_3_var,\r\n        round(abs(power(10, AVG(log(10, t1.ic50))\r\n                      OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant,\r\n                                        t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier))* 1000000000 \r\n                                        + (3 * VARIANCE(t1.ic50)\r\n                                                                                                OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.\r\n                                                                                                cell_line, t1.variant, t1.cell_incubation_hr,\r\n                                                                                                t1.pct_serum,t1.threed, t1.treatment, t1.modifier) * 1000000000)), 3) AS nM_plus_3_var,\r\n        count(t1.ic50) OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier) AS n,\r\n        count(t1.ic50) OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.threed, t1.treatment, t1.pct_serum) AS m,\r\nt1.cro || '|' || t1.assay_type || '|' || t1.cell_line || '|' || nvl(t1.variant, '-') || '|' || t1.cell_incubation_hr || '|' || t1.pct_serum || '|' || case t1.threed when 'Y' then '3D' else '-' end || '|' || case when t1.treatment is not null then treatment else '-' end || '|' || t1.modifier AS cellvalue_two\r\n    FROM\r\n        ds3_userdata.su_cellular_growth_drc t1\r\n        LEFT OUTER JOIN ds3_userdata.CELLULAR_IC50_FLAGS t2 ON t1.PID = t2.PID\r\nWHERE\r\n        t1.assay_intent = 'Screening'\r\n        AND t1.validated = 'VALIDATED'\r\n        AND trim(t1.washout) = 'N'\r\n) t0\r\nGROUP BY\r\n    t0.compound_id,\r\n    t0.cro,\r\n    t0.assay_type,\r\n    t0.cell_line,\r\n    t0.variant,\r\n    t0.cell_incubation_hr,\r\n    t0.pct_serum,\r\n    t0.threed,\r\n    t0.treatment,\r\n    t0.flag,\r\n    t0.modifier\r\nORDER BY \r\n    t0.compound_id,\r\n    t0.cro,\r\n    t0.assay_type,\r\n    t0.cell_line,\r\n    t0.variant,\r\n    t0.cell_incubation_hr,\r\n    t0.pct_serum,\r\n    t0.threed,\r\n    t0.treatment"
"""

test_sql_str = test_sql_str.replace("\r", "").replace("\n", "")
parsed_str = sqlparse.parse(test_sql_str)[0]

# subquery_select_lines, subquery_from_clause = extract_subquery(
#     parsed_str, subquery_cell_geomean
# )
# print("----")
# print(subquery_from_clause)


# d = extract_select_expr(parsed_str, subquery_cell_geomean)
# print(d)

# main_select_expr = "\n,".join(main_select_lines)
# subquery_select_expr = "\n,".join(subquery_select_lines)

# sql_string = f"""
#     SELECT
#     {main_select_expr}
#     ,TO_CHAR(max(t0.created_date)) as created_date
#     ,max(t0.date_highlight) as date_highlight
#     FROM (
#     SELECT
#     {subquery_select_expr}
#     ,created_date
#     {subquery_from_clause}
#     {subquery_where_clause}
#     ) t0
#     {where_clause} and t0.compound_id = '{{0}}'
#     {group_by_expression}
#     ORDER BY CREATED_DATE DESC
# """

# print(sql_string.format("FT007615"))

# parsed_str._pprint_tree()
