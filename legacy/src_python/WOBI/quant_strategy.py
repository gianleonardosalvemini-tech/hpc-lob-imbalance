import os
import ctypes
import pandas as pd
import numpy as np

# --- STRUTTURE C (Manteniamo quelle che hanno appena funzionato) ---
class SignalMetrics(ctypes.Structure):
    _fields_ = [
        ("weighted_micro_price", ctypes.c_double),
        ("imbalance_pressure", ctypes.c_double)
    ]

class LOBLevel(ctypes.Structure):
    _fields_ = [("price", ctypes.c_double), ("volume", ctypes.c_double)]

class LOBState(ctypes.Structure):
    _fields_ = [
        ("bids", LOBLevel * 10),
        ("asks", LOBLevel * 10)
    ]

# --- CARICAMENTO DLL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.abspath(os.path.join(current_dir, "..", "..", "src_c", "WOBI", "wobi_engine.dll"))

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(os.path.dirname(dll_path))

lob_lib = ctypes.CDLL(dll_path)
lob_lib.compute_signals.restype = SignalMetrics
lob_lib.compute_signals.argtypes = [ctypes.POINTER(LOBState), ctypes.c_double]


def run_backtest(csv_path, alpha=0.5, entry_threshold=0.2, taker_fee=0.0004, wobi_threshold=0.6, max_hold_time_ms=2500):
    print(f"Caricamento dataset: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    state = LOBState()
    trades = []
    
    # Gestione dello stato della posizione
    active_trade = None
    
    print(f"Avvio simulazione HPC con Gestione del Rischio (Time-Stop: {max_hold_time_ms}ms)...")
    
    for row in df.itertuples(index=False):
        timestamp = int(row[1])
        
        for i in range(10):
            state.bids[i].price = float(row[3 + 2*i])
            state.bids[i].volume = float(row[4 + 2*i])
            state.asks[i].price = float(row[23 + 2*i])
            state.asks[i].volume = float(row[24 + 2*i])
            
        metrics = lob_lib.compute_signals(ctypes.byref(state), alpha)
        
        best_bid = state.bids[0].price
        best_ask = state.asks[0].price
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        fair_value_delta = metrics.weighted_micro_price - mid_price
        
        # ---------------------------------------------------------
        # GESTIONE POSIZIONE APERTA (EXIT STRATEGY)
        # ---------------------------------------------------------
        if active_trade is not None:
            time_elapsed = timestamp - active_trade['entry_time']
            exit_price = 0.0
            exit_reason = None
            
            if active_trade['type'] == 'LONG':
                # Condizione 1: Take Profit Dinamico (raggiunto il Fair Value target)
                if best_bid >= active_trade['tp_target']:
                    exit_price = best_bid - (spread * 0.1) # Slippage in uscita
                    exit_reason = 'TP'
                # Condizione 2: Stop Loss Microstrutturale
                elif best_bid <= active_trade['sl_target']:
                    exit_price = best_bid - (spread * 0.1)
                    exit_reason = 'SL'
                # Condizione 3: Time-Stop (Decadimento del Segnale EV)
                elif time_elapsed > max_hold_time_ms:
                    exit_price = best_bid - (spread * 0.1)
                    exit_reason = 'TIME_STOP'
                    
            elif active_trade['type'] == 'SHORT':
                if best_ask <= active_trade['tp_target']:
                    exit_price = best_ask + (spread * 0.1)
                    exit_reason = 'TP'
                elif best_ask >= active_trade['sl_target']:
                    exit_price = best_ask + (spread * 0.1)
                    exit_reason = 'SL'
                elif time_elapsed > max_hold_time_ms:
                    exit_price = best_ask + (spread * 0.1)
                    exit_reason = 'TIME_STOP'
            
            # Se una condizione di uscita è scattata, chiudiamo il trade
            if exit_reason:
                pnl = (exit_price - active_trade['entry_price']) if active_trade['type'] == 'LONG' else (active_trade['entry_price'] - exit_price)
                # Sottraiamo le taker fee in uscita
                pnl -= (exit_price * taker_fee)
                
                active_trade['exit_time'] = timestamp
                active_trade['exit_price'] = exit_price
                active_trade['exit_reason'] = exit_reason
                active_trade['pnl_net'] = pnl
                active_trade['hold_time'] = time_elapsed
                
                trades.append(active_trade)
                active_trade = None # Resetta lo stato per cercare nuovi ingressi
                
            continue # Salta la logica di ingresso se siamo già a mercato o stiamo uscendo

        # ---------------------------------------------------------
        # LOGICA DI INGRESSO (ENTRY STRATEGY)
        # ---------------------------------------------------------
        cost_of_trading = spread + (mid_price * taker_fee)
        
        # Trigger LONG
        if fair_value_delta > (entry_threshold + cost_of_trading) and metrics.imbalance_pressure > wobi_threshold:
            execution_price = best_ask + (spread * 0.1)
            active_trade = {
                'entry_time': timestamp,
                'type': 'LONG',
                'entry_price': execution_price + (execution_price * taker_fee), # Includiamo fee in ingresso
                'wobi': metrics.imbalance_pressure,
                'tp_target': metrics.weighted_micro_price, # Il target è il Fair Value calcolato
                'sl_target': execution_price - (spread * 3) # SL a 3 spread di distanza
            }
            
        # Trigger SHORT
        elif fair_value_delta < -(entry_threshold + cost_of_trading) and metrics.imbalance_pressure < -wobi_threshold:
            execution_price = best_bid - (spread * 0.1)
            active_trade = {
                'entry_time': timestamp,
                'type': 'SHORT',
                'entry_price': execution_price - (execution_price * taker_fee),
                'wobi': metrics.imbalance_pressure,
                'tp_target': metrics.weighted_micro_price,
                'sl_target': execution_price + (spread * 3)
            }

    print(f"Backtest completato. Trade chiusi: {len(trades)}")
    return pd.DataFrame(trades)

if __name__ == "__main__":
    # Inserimento del percorso assoluto esatto
    dataset_path = r"C:\Users\gianl\Documents\hpc-lob-imbalance\src_python\1-09-1-20.csv"
    
    if os.path.exists(dataset_path):
        results_df = run_backtest(dataset_path)
        if not results_df.empty:
            print(results_df.head()) 
    else:
        print(f"ATTENZIONE: Dataset non trovato in {dataset_path}") 