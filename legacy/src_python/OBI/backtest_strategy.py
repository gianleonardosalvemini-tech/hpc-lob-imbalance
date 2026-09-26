import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_parameter_sweep(filename, nrows=100000):
    print(f"Caricamento di {nrows} tick dal dataset...")
    df = pd.read_csv(filename, nrows=nrows)

    # Estrazione colonne LOB
    bid_p = df.iloc[:, 3].astype(float).values
    bid_v = df.iloc[:, 4].astype(float).values
    ask_p = df.iloc[:, 23].astype(float).values
    ask_v = df.iloc[:, 24].astype(float).values

    obi = (bid_v - ask_v) / (bid_v + ask_v)

    # Parametri da testare
    threshold_obi = 0.6  # Soglia di squilibrio per generare il segnale
    holding_horizons = [4, 20, 80]  # Orizzonte in tick (1s, 5s, 20s a 250ms)
    slippage_levels = [0.0, 0.5, 1.0, 2.0]  # Costo per trade in USDT (latenza addizionale)

    results = []

    print("\n--- AVVIO SWEEP DEI PARAMETRI (CORRETTO PER MICROSTRUTTURA) ---")
    for horizon in holding_horizons:
        
        # PnL Reale: incrocio dello spread (Market Taking puro)
        # Se andiamo Long: Compriamo all'Ask al tempo t, Vendiamo al Bid al tempo t+horizon
        raw_pnl_long_array = np.roll(bid_p, -horizon) - ask_p
        
        # Se andiamo Short: Vendiamo al Bid al tempo t, Ricompriamo all'Ask al tempo t+horizon
        raw_pnl_short_array = bid_p - np.roll(ask_p, -horizon)
        
        # Escludiamo i trade che finirebbero fuori dal dataset
        valid_mask = np.arange(len(obi)) < (len(obi) - horizon)

        # Inizializzazione variabili per il Cooldown
        in_position_until = 0
        filtered_long_indices = []
        filtered_short_indices = []

        # Estrazione dei segnali statisticamente indipendenti
        for i in range(len(obi)):
            if i < in_position_until or not valid_mask[i]:
                continue
            
            if obi[i] > threshold_obi:
                filtered_long_indices.append(i)
                in_position_until = i + horizon  # Il sistema non accetta nuovi segnali finché il trade è aperto
            elif obi[i] < -threshold_obi:
                filtered_short_indices.append(i)
                in_position_until = i + horizon

        raw_pnl_long = raw_pnl_long_array[filtered_long_indices]
        raw_pnl_short = raw_pnl_short_array[filtered_short_indices]
        total_trades = len(raw_pnl_long) + len(raw_pnl_short)

        if total_trades == 0:
            continue

        raw_pnl = np.concatenate([raw_pnl_long, raw_pnl_short])

        for slip in slippage_levels:
            # Sottraiamo solo eventuale slippage per latenza (lo spread è già calcolato nel PnL raw)
            net_pnl = raw_pnl - slip
            win_rate = np.mean(net_pnl > 0) * 100
            total_net_profit = np.sum(net_pnl)
            ev_per_trade = np.mean(net_pnl)

            results.append({
                'Horizon (ticks)': horizon,
                'Horizon (sec)': horizon * 0.25,
                'Slippage (USDT)': slip,
                'Trades': total_trades,
                'Win Rate (%)': win_rate,
                'EV/Trade (USDT)': ev_per_trade,
                'Total Net PnL': total_net_profit
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Grafico
    plt.figure(figsize=(10, 6))
    for horizon in holding_horizons:
        subset = results_df[results_df['Horizon (ticks)'] == horizon]
        plt.plot(subset['Slippage (USDT)'], subset['EV/Trade (USDT)'], marker='o', 
                 label=f"Orizzonte: {horizon} tick ({horizon*0.25:.1f}s)")

    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("Expected Value per Trade: Impatto Latenza e Orizzonte (Spread Inclusa)", fontweight='bold')
    plt.xlabel("Costo Slippage Latenza (USDT)")
    plt.ylabel("EV Netto per Trade (USDT)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pnl_decay_analysis.png", dpi=300)
    print("\nGrafico salvato come 'pnl_decay_analysis.png'.")
    plt.show()

if __name__ == "__main__":
    run_parameter_sweep("1-09-1-20.csv", nrows=100000)