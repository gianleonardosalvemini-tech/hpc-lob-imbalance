import ctypes
import os
import csv
import time

# 1. Mappatura delle strutture C
class LOBLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double),
                ("volume", ctypes.c_int)]

class LOBState(ctypes.Structure):
    _fields_ = [("timestamp", ctypes.c_longlong),
                ("best_bid", LOBLevel),
                ("best_ask", LOBLevel)]

# 2. Caricamento del motore C (.dll)
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src_c', 'lob_engine.dll'))
lob_engine = ctypes.CDLL(lib_path)
lob_engine.init_lob_state.argtypes = [ctypes.POINTER(LOBState), ctypes.c_longlong, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lob_engine.compute_raw_obi.argtypes = [ctypes.POINTER(LOBState)]
lob_engine.compute_raw_obi.restype = ctypes.c_double

def process_real_market_data(filename):
    current_market = LOBState()
    valid_signals = 0
    total_ticks = 0
    
    print(f"Avvio backtest quantitativo sui dati reali ({filename})...")
    start_time = time.time()
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader) # Salta la riga di intestazione numerica
        
        for row in reader:
            try:
                # Nuova mappatura basata sull'indagine critica della struttura:
                timestamp_micro = int(row[1])
                
                # Best Bid ai nuovi indici 3 (Prezzo) e 4 (Volume)
                best_bid_p = float(row[3])
                best_bid_v = int(float(row[4])) 
                
                # Best Ask ai nuovi indici 23 (Prezzo) e 24 (Volume)
                best_ask_p = float(row[23])
                best_ask_v = int(float(row[24]))
                
                total_ticks += 1
                
                # A. Ottimizzazione Hardware: Aggiornamento memoria LOB in C
                lob_engine.init_lob_state(ctypes.byref(current_market), timestamp_micro, best_bid_p, best_bid_v, best_ask_p, best_ask_v)
                obi_signal = lob_engine.compute_raw_obi(ctypes.byref(current_market))
                
                # B. Analisi EV e Rischio (Fattore Anti-Tutorial)
                prob_success = 0.50 + (abs(obi_signal) * 0.20)
                prob_loss = 1.0 - prob_success
                slippage_cost = 0.5 
                
                expected_value = (prob_success * (2 - slippage_cost)) - (prob_loss * (1 + slippage_cost))
                
                if expected_value > 0.1:
                    valid_signals += 1

            except (ValueError, IndexError):
                # Filtra le righe di mercato corrotte senza far crashare il motore
                continue 

    execution_time = time.time() - start_time
    print(f"Backtest completato in {execution_time:.4f} secondi.")
    print(f"Tick di mercato elaborati: {total_ticks}")
    if total_ticks > 0:
        print(f"Segnali profittevoli netti post-slippage: {valid_signals} ({valid_signals/total_ticks*100:.2f}%)")

if __name__ == "__main__":
    csv_file = "1-09-1-20.csv"
    if os.path.exists(csv_file):
        process_real_market_data(csv_file)
    else:
        print(f"Errore: Il file {csv_file} non si trova nella cartella src_python.")