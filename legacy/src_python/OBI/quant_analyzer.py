import ctypes
import os

# 1. Mappatura delle strutture C in Python
class LOBLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double),
                ("volume", ctypes.c_int)]

class LOBState(ctypes.Structure):
    _fields_ = [("timestamp", ctypes.c_longlong),
                ("best_bid", LOBLevel),
                ("best_ask", LOBLevel)]

# 2. Caricamento della libreria dinamica (.dll) compilata in C
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src_c', 'lob_engine.dll'))
lob_engine = ctypes.CDLL(lib_path)

# 3. Configurazione dei parametri per le funzioni C
lob_engine.init_lob_state.argtypes = [ctypes.POINTER(LOBState), ctypes.c_longlong, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
lob_engine.compute_raw_obi.argtypes = [ctypes.POINTER(LOBState)]
lob_engine.compute_raw_obi.restype = ctypes.c_double

# --- TEST DI ESECUZIONE CON ANALISI EV ---
if __name__ == "__main__":
    current_market = LOBState()
    
    # Simuliamo il tick che hai appena testato (Bid=500, Ask=100)
    lob_engine.init_lob_state(ctypes.byref(current_market), 1679483000, 100.50, 500, 100.55, 100)
    obi_signal = lob_engine.compute_raw_obi(ctypes.byref(current_market))
    
    print(f"Segnale OBI puro (C Engine): {obi_signal:.4f}")
    
    # 4. Modello di Rischio: Expected Value 
    # Se OBI è 0.66, c'è forte pressione in acquisto, ma qual è il vero EV?
    
    take_profit_ticks = 2  # Puntiamo a guadagnare 2 tick (es. da 100.55 a 100.65)
    stop_loss_ticks = 1    # Tagliamo le perdite dopo 1 tick contro di noi
    
    # La probabilità di successo scala con l'intensità del segnale OBI
    prob_success = 0.50 + (obi_signal * 0.20) 
    prob_loss = 1.0 - prob_success
    
    # Modello di latenza e slippage (i profitti teorici vengono erosi dal mercato reale)
    slippage_cost = 0.5 # Costo in tick perso per latenza di esecuzione
    
    # Calcolo EV: (Prob_Vittoria * Guadagno_Netto) - (Prob_Sconfitta * Perdita_Netta)
    expected_value = (prob_success * (take_profit_ticks - slippage_cost)) - (prob_loss * (stop_loss_ticks + slippage_cost))
    
    print(f"Probabilità stimata di rialzo a breve termine: {prob_success * 100:.1f}%")
    print(f"Expected Value (EV) per trade in tick: {expected_value:.4f}")
    
    if expected_value > 0.1:
        print("Esito: SEGNALE VALIDO. L'edge copre i costi di slippage.")
    else:
        print("Esito: FALSO POSITIVO. Il segnale decade prima dell'esecuzione.")