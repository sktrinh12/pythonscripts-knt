from com.dotmatics.dataig.studies.dataparser.processor import ExcelFileProcessor
from com.dotmatics.dataig.studies.dataparser.builder import TableBuilder
from com.dotmatics.dataig.studies.dataparser.data import Row

# parse excel file
fp = ExcelFileProcessor(data.getFile())
f = fp.process()

# define table
table1 = TableBuilder.build("FT_PHARM_STUDY")

tableCols = [
    "STUDY_ID",
    "PROJECT_ID",
    "REQUESTOR",
    "STUDY_DIRECTOR",
    "TEAM_ID",
    "PROTOCOL_TYPE",
    "PROTOCOL_ID",
    "REQUEST_DATE",
    "IN_LIFE_DATE",
    "REPORT_DATE",
    "NOTES",
    "IMAGE_FILE",
]
# imageFileColumn = table1.getColumnByName('NOTES')


# get sheet 1
sheet1 = f.getSheetByName("COVER")

# add data block 1
data_block1 = data.addDataBlock("Block 1", table1)

for i, r in enumerate(range(1, sheet1.getNumRows(), 1)):
    row = Row(i + 1)
    data_block1.addRow(row)

    # loop through each column in input file
    for c in range(0, len(tableCols), 1):
        # row.addCell(col_name, value)
        row.addCell(table1.getColumnByName(tableCols[c]), sheet1.getCellValue(r, c))

    # get sheet 2
sheet2 = f.getSheetByName("GROUP")
table2 = TableBuilder.build("FT_PHARM_GROUP")

# add data block 2
data_block2 = data.addDataBlock("Block 2", table2)

# define table col names - these must match the db table col names!

tableCols = [
    "GROUP_ID",
    "ROUTE",
    "FREQUENCY",
    "N_DOSING",
    "SUBJECT_TYPE",
    "SUBJECT_ID",
    "DOSE_ID",
    "DOSE",
    "DOSE_UNIT",
    "DOSING_SITE",
    "IS_FED",
]
# loop through each row in input file
# skip first 2 header rows in this example**
for i, r in enumerate(range(1, sheet2.getNumRows(), 1)):
    row = Row(i + 1)
    data_block2.addRow(row)

    # loop through each column in input file
    for c in range(0, len(tableCols), 1):
        # row.addCell(col_name, value)
        row.addCell(table2.getColumnByName(tableCols[c]), sheet2.getCellValue(r, c))


# get sheet 3
sheet3 = f.getSheetByName("DOSE")
table3 = TableBuilder.build("FT_PHARM_DOSE")

# add data block 3
data_block3 = data.addDataBlock("Block 3", table3)

# define table col names - these must match the db table col names!

tableCols = [
    "DOSE_ID",
    "BATCH_ID",
    "FORMULATION_ID",
    "CONCENTRATION",
    "CONCENTRATION_UNIT",
    "TREATMENT",
    "APPEARANCE",
    "COMMENTS",
    "MEAN_CONC",
    "ACCURACY_PCT",
    "SD",
]
# loop through each row in input file
# skip first 2 header rows in this example**
for i, r in enumerate(range(1, sheet3.getNumRows(), 1)):
    row = Row(i + 1)
    data_block3.addRow(row)

    # loop through each column in input file
    for c in range(0, len(tableCols), 1):
        # row.addCell(col_name, value)
        row.addCell(table3.getColumnByName(tableCols[c]), sheet3.getCellValue(r, c))


# get sheet 4
sheet4 = f.getSheetByName("DOSING")
table4 = TableBuilder.build("FT_PHARM_DOSING")

# add data block 4
data_block4 = data.addDataBlock("Block 4", table4)

# define table col names - these must match the db table col names!

tableCols = [
    "DOSING_ID",
    "SUBJECT_ID",
    "AMOUNT",
    "AMOUNT_UNIT",
    "DOSING_TIME",
    "DOSING_TIME_UNIT",
]
# loop through each row in input file
# skip first 2 header rows in this example**
for i, r in enumerate(range(1, sheet4.getNumRows(), 1)):
    row = Row(i + 1)
    data_block4.addRow(row)

    # loop through each column in input file
    for c in range(0, len(tableCols), 1):
        # row.addCell(col_name, value)
        row.addCell(table4.getColumnByName(tableCols[c]), sheet4.getCellValue(r, c))

# get sheet 5
# sheet1 = f.getSheetAtIndex(0)
sheet5 = f.getSheetByName("TVBW")
# define table
table5 = TableBuilder.build("FT_PHARM_EFFICACY_RAW")

# add data block 1
data_block5 = data.addDataBlock("Block 5", table5)

# define table col names - these must match the db table col names!
# tableCols = ['SUPPLIER_REF', 'FORMATTED_SAMPLE_ID', 'BARCODE', 'ASSAY_NAME', 'RESULT_TYPE', 'CONC_UNIT',
# 'RESULT_NUMERIC', 'RESULT_MODIFIER', 'RESULT_ALPHA', 'VALIDATED', 'COMMENTS']
tableCols = [
    "SAMPLE_ID",
    "ANIMAL_ID",
    "SAMPLING_TIME",
    "BW_G",
    "L_MM",
    "W_MM",
    "TV_MM_3",
    "OBSERVATION",
]
for i, r in enumerate(range(1, sheet5.getNumRows(), 1)):
    row = Row(i + 1)
    data_block5.addRow(row)
    for c in range(0, len(tableCols), 1):
        # row.addCell(col_name, value)
        row.addCell(table5.getColumnByName(tableCols[c]), sheet5.getCellValue(r, c))

### TESTING
import decimal

# Define the table columns
tableCols = [
    "SAMPLE_ID",
    "ANIMAL_ID",
    "SAMPLING_TIME",
    "BW_G",
    "L_MM",
    "W_MM",
    "TV_MM_3",
    "OBSERVATION",
]


# Loop through the rows in the Excel file
for i, r in enumerate(range(1, sheet5.getNumRows(), 1)):
    row = Row(i + 1)
    data_block5.addRow(row)
    for c in range(0, len(tableCols), 1):
        # Convert floating point values to decimal
        if tableCols[c] in ["BW_G", "L_MM", "W_MM", "TV_MM_3"]:
            cell_value = decimal.Decimal(str(sheet5.getCellValue(r, c)))
        else:
            cell_value = sheet5.getCellValue(r, c)
        # Add the cell value to the row
        row.addCell(table5.getColumnByName(tableCols[c]), cell_value)
