import json

notebook_name = "AAPL_Prediction_corrected.ipynb"

with open(notebook_name, encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"], start=1):
    print()
    print("=" * 80)
    print(f"CELL {i} [{cell['cell_type'].upper()}]")
    print("=" * 80)
    print("".join(cell.get("source", [])))