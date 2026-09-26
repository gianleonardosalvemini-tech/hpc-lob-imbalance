import pandas as pd
import matplotlib.pyplot as plt

def plot_lob_imbalance(filename, num_ticks=5000):
    print(f"Estrazione visiva di una finestra di alta volatilità ({num_ticks} tick)...")
    
    # Carichiamo solo un frammento temporale per rendere il grafico analizzabile a occhio nudo
    df = pd.read_csv(filename, nrows=num_ticks)
    
    # Utilizziamo la mappatura esatta che abbiamo scoperto con l'indagine critica
    bid_prices = df.iloc[:, 3].astype(float)
    bid_vols = df.iloc[:, 4].astype(float)
    ask_prices = df.iloc[:, 23].astype(float)
    ask_vols = df.iloc[:, 24].astype(float)
    
    # Calcoliamo il Mid-Price (il prezzo medio tra la migliore offerta di acquisto e vendita)
    mid_price = (bid_prices + ask_prices) / 2
    
    # Calcoliamo l'Order Book Imbalance (OBI)
    obi = (bid_vols - ask_vols) / (bid_vols + ask_vols)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Asse sinistro: Andamento del Prezzo
    ax1.set_xlabel('Tick (Campionamento a 250ms)')
    ax1.set_ylabel('Mid Price (USDT)', color='#1f77b4', fontweight='bold')
    ax1.plot(mid_price, color='#1f77b4', alpha=0.9, linewidth=1.5, label='Mid Price')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    # Asse destro: Squilibrio del Portafoglio Ordini (OBI)
    ax2 = ax1.twinx()
    ax2.set_ylabel('LOB Imbalance (OBI)', color='#d62728', fontweight='bold')
    ax2.plot(obi, color='#d62728', alpha=0.3, linewidth=0.8, label='OBI Signal')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Linea dello zero (Equilibrio perfetto tra compratori e venditori)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2)
    
    plt.title('Critica del Modello: Limit Order Book Imbalance vs Price Action', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_lob_imbalance("1-09-1-20.csv", 5000)

    