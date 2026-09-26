# High-Frequency LOB Imbalance: C/Python Engine & Microstructure Analysis

## Il Progetto in Sintesi
L'obiettivo iniziale di questo progetto era testare se l'Order Book Imbalance (OBI) potesse essere usato come segnale direzionale profittevole su BTC/USDT di Binance (tick-by-tick, campionamento 250ms). 

Lavorando con dataset di queste dimensioni (oltre 3.7 milioni di righe per una manciata di giorni), mi sono subito scontrato con i limiti fisici di Python: caricare e iterare tutto in Pandas dilatava i tempi di test in modo inaccettabile. Ho quindi optato per un'architettura ibrida.

## 1. Architettura Hardware e Software
Per non avere colli di bottiglia, ho separato il *number crunching* dall'analisi statistica:
*   **HPC Engine in C:** Ho scritto la logica di aggiornamento del Limit Order Book in puro C. Gestire manualmente l'aritmetica dei puntatori mi ha permesso di valorizzare l'hardware (sfruttando il multithreading del mio Intel Core Ultra 7) ed elaborare i 3.7 milioni di tick a bassissima latenza in circa 22 secondi.
*   **Python Wrapper:** Tramite `ctypes`, ho richiamato la libreria C precompilata in Python. In questo modo mantengo la velocità di esecuzione del C, ma posso usare `pandas` e `matplotlib` per analizzare i segnali senza dover ricompilare tutto a ogni cambio di parametro.

## 2. Il "Miraggio" del 90% di Win Rate
Il motore calcola dinamicamente lo squilibrio tra i volumi al Best Bid e Best Ask:
`OBI = (Volume_Bid - Volume_Ask) / (Volume_Bid + Volume_Ask)`

Ho impostato un trigger base: se `|OBI| > 0.6`, il mercato è fortemente sbilanciato e parte un ordine *Market* nella direzione del volume.
Il primo backtest "ingenuo" ha restituito un win rate superiore al 90%. Ovviamente, era un risultato irrealistico. 

## 3. Critica dei Risultati (Perché il modello direzionale fallisce)
Ho iniziato a calcolare l'Expected Value (EV) dei trade soppesando i rendimenti lordi contro i costi reali di *slippage*. Applicando la stessa rigorosa forma mentis analitica utilizzata per calcolare le *pot odds* in una mano di Texas Hold'em, l'illusione si è subito sgretolata.

Come si nota nel grafico `Figure_1.png` generato dall'analisi, l'OBI (linea rossa) satura costantemente verso +1.0 o -1.0, ma il *mid-price* (linea blu) spesso rimane bloccato in congestioni strettissime. 

**L'illusione del Take Profit e l'Alpha Decay**
Un'obiezione logica potrebbe essere: *perché non allungare semplicemente il Take Profit per dare al prezzo il tempo di muoversi e coprire i costi di spread?*
L'analisi matematica smentisce questa ipotesi. L'OBI è un misuratore di pressione istantanea, non di trend. Eseguendo un *parameter sweep* su orizzonti temporali multipli (`pnl_decay_analysis.png`), l'Alpha Decay emerge chiaramente: allungando il tempo a mercato per inseguire un target più ampio, il vantaggio statistico iniziale svanisce rapidamente, divorato dal rumore stocastico. 

Su un orizzonte brevissimo di 4 tick (1 secondo), basta applicare uno *slippage* di appena 0.3 USDT (spread + latenza) per portare l'Expected Value netto in territorio negativo. In pratica, il prezzo non si muove abbastanza velocemente per coprire il costo di attraversamento dello spread prima che il segnale decada.

## 4. La Soluzione Quantitativa: Dal Market Taking al Market Making
I dati mi hanno dimostrato che l'OBI è un indicatore microstrutturale valido, ma usarlo per "aggredire" il mercato (Market Taking) è matematicamente perdente a causa dei costi di transazione.

La soluzione ingegneristica è stata ribaltare la logica di esecuzione: usare l'OBI per il **Market Making asimmetrico**.
Invece di lanciare ordini a mercato quando l'imbalance sale oltre 0.6 pagando lo spread, il sistema usa il segnale per piazzare ordini *Limit* passivi sul lato forte del book. Fornendo liquidità, non paghiamo più lo slippage, ma ci mettiamo in posizione per incassare lo spread noi stessi, limitando il rischio direzionale avverso.
