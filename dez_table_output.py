import texttable

x = texttable.Texttable()
x.add_rows(
    [
        ["RECORD_DATE", "RECORD_EXTRACTED", "ID"],
        ["1676271601000", "null", 1],
        ["1677394801000", "null", 2],
        ["1677481201000", "null", 3],
        ["1677567601000", "null", 4],
    ]
)
print(x.draw())
