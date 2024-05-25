from openpyxl import load_workbook

def handle_xlsx():
    wb = load_workbook()
    # Select the first sheet
    sheet = wb.active
    temp=""
    # Iterate over all cells in the sheet
    for row in sheet.iter_rows():
        for cell in row:
            temp += str(cell.value) + " " if cell.value is not None else ""

    print(temp)