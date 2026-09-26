import csv
dataset_path = r"C:\Users\gianl\Documents\hpc-lob-imbalance\src_python\1-09-1-20.csv"
with open(dataset_path, "r") as file:
    reader = csv.reader(file)
    headers = next(reader)
    
    print("--- STRUTTURA REALE DEL DATASET ---")
    print(f"Colonna 0: {headers[0]}")
    print(f"Colonna 1: {headers[1]}")
    print(f"Colonna 2 (Ipotesi Best Bid Price): {headers[2]}")
    print(f"Colonna 22 (Ipotesi Best Ask Price): {headers[22]}")
    print(f"Totale colonne rilevate: {len(headers)}")