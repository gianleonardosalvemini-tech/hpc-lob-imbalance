import ctypes
import os
import random
import time

# 1. Mappatura delle strutture C
class LOBLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double),
                ("volume", ctypes.c_int)]

class LOBState(ctypes.Structure):
    _fields_ = [("timestamp", ctypes.c_longlong),
                ("best_bid", LOBLevel),
                ("best_ask", LOBLevel)]

# 2. Caricamento della libreria dinamica (.dll)
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src_c', 'lob_engine.dll'))
lob_engine = ctypes.CDLL(lib_path)

# 3. Configurazione dei parametri
lob_engine.init_lob_state.argtypes = [ctypes.POINTER(LOBState), ctypes.c_longlong, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lob_engine.compute_raw_obi.argtypes = [ctypes.POINTER(LOBState)]
lob_engine.compute_raw_obi.restype = ctypes.c_double

# --- STRESS TEST E ANALISI EV ---
if __name__ == "__main__":
    current_market = LOBState()
    num_ticks = 100000  # Simulazione di 100.000 aggiornamenti del book
    valid_signals = 0
    
    print(f"Avvio stress test su {num_ticks} tick...")
    start_time = time.time() # Facciamo partire il cronometro
    
    for i in range(num_ticks):
        # Generazione volumi casuali per simulare il caos del mercato
        bid_vol = random.randint(10, 1000)
        ask_vol = random.randint(10, 1000)
        
        # A. Motore C: Aggiornamento memoria a bassissima latenza
        lob_engine.init_lob_state(ctypes.byref(current_market), int(time.time() * 1000), 100.50, bid_vol, 100.55, ask_vol)
        obi_signal = lob_engine.compute_raw_obi(ctypes.byref(current_market))
        
        # B. Modello di Rischio Python: Valutazione EV e decadimento
        prob_success = 0.50 + (abs(obi_signal) * 0.20) 
        prob_loss = 1.0 - prob_success
        slippage_cost = 0.5 # Costo in tick perso per latenza
        
        # EV = (Prob_Vittoria * Guadagno_Netto) - (Prob_Sconfitta * Perdita_Netta)
        expected_value = (prob_success * (2 - slippage_cost)) - (prob_loss * (1 + slippage_cost))
        
        # Contiamo quanti segnali sopravvivono ai costi di esecuzione
        if expected_value > 0.1:
            valid_signals += 1

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Simulazione completata in {execution_time:.4f} secondi.")
    print(f"Segnali validi (EV > 0.1) sopravvissuti allo slippage: {valid_signals} su {num_ticks} ({valid_signals/num_ticks*100:.2f}%)")