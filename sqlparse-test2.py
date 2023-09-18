import sqlparse


def get_query_columns(sql):
    stmt = sqlparse.parse(sql)[0]
    columns = []
    column_identifiers = []

    in_select = False
    for token in stmt.tokens:
        if isinstance(token, sqlparse.sql.Comment):
            continue
        if str(token).lower() == "select":
            in_select = True
        elif in_select and token.ttype is None:
            for identifier in token.get_identifiers():
                column_identifiers.append(identifier)
            break
    for column_identifier in column_identifiers:
        columns.append(column_identifier.get_name())

    return columns


def test2():
    sql = """
SELECT
        t1.cro,
        t1.project,
        t1.assay_type,
        t1.compound_id,
        t1.batch_id,
        t1.cell_line,
        nvl(t1.variant, '-') AS variant,
        t1.created_date, 
        t1.cell_incubation_hr,
        t1.pct_serum,
        case t1.threed when 'Y' then '3D' else '-' end AS threed,
        case when t1.treatment is not null then treatment else '-' end AS treatment,
        t1.modifier,
        ROUND((TO_CHAR(POWER(10, AVG(LOG(10, t1.ic50)) 
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier)), '99999.99EEEE') * 1000000000), 1) AS geomean_nM,
        COUNT(t1.ic50) 
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.pct_serum, t1.threed, t1.treatment, t1.modifier) AS n,
        COUNT(t1.ic50) 
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.cell_line, t1.variant, t1.cell_incubation_hr, t1.threed, t1.treatment, t1.pct_serum) AS m
    FROM
        ds3_userdata.su_cellular_growth_drc t1
"""
    print(get_query_columns(sql))


test2()


def test3():
    sql = """SELECT
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
            COUNT(t1.ic50)
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,
                              t1.cofactors, t1.atp_conc_um, t1.modifier)               AS n,
            COUNT(t1.ic50)
            OVER(PARTITION BY t1.compound_id, t1.cro, t1.assay_type, t1.target, t1.variant,
                              t1.cofactors, t1.atp_conc_um)                            AS m
        FROM
            ds3_userdata.su_biochem_drc t1
        WHERE
                assay_intent = 'Screening'
            AND validated = 'VALIDATED'
"""
    print(get_query_columns(sql))


test3()
