//definizione delle strutture di alocazione della memoria del LOB
#include <stdio.h>
#include <stdlib.h>

// Struttura per un singolo livello di prezzo (Bid o Ask)
typedef struct {
    double price;
    int volume;
} LOBLevel;

// Struttura per l'istantanea del portafoglio ordini
typedef struct {
    long long timestamp; // Tempo in millisecondi
    LOBLevel best_bid;   // Lato acquisti
    LOBLevel best_ask;   // Lato vendite
} LOBState;

// Funzione base per inizializzare lo stato del book
void init_lob_state(LOBState* state, long long ts, double bid_p, int bid_v, double ask_p, int ask_v) {
    state->timestamp = ts;
    state->best_bid.price = bid_p;
    state->best_bid.volume = bid_v;
    state->best_ask.price = ask_p;
    state->best_ask.volume = ask_v;
}

// Funzione per aggiornare lo stato del LOB all'arrivo di un nuovo tick ad alta frequenza
void update_lob_state(LOBState* state, long long ts, double bid_p, int bid_v, double ask_p, int ask_v) {
    state->timestamp = ts;
    
    // Aggiornamento lato Bid (Domanda)
    state->best_bid.price = bid_p;
    state->best_bid.volume = bid_v;
    
    // Aggiornamento lato Ask (Offerta)
    state->best_ask.price = ask_p;
    state->best_ask.volume = ask_v;
}

// Funzione preliminare per calcolare lo squilibrio grezzo (Order Book Imbalance) (OBI)
double compute_raw_obi(const LOBState* state) {
    int bid_vol = state->best_bid.volume;
    int ask_vol = state->best_ask.volume;
    
    // Evitiamo divisioni per zero in caso di book anomalo
    if ((bid_vol + ask_vol) == 0) {
        return 0.0;
    }
    
    // Formula standard dell'Imbalance: (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
    return (double)(bid_vol - ask_vol) / (bid_vol + ask_vol);
}
//la parola chiave const (costante) indica che la funzione si limiterà 
// a leggere i dati dello stato del Limit Order Book senza poterli modificare per errore

//Efficienza Hardware (HPC): Passando un puntatore costante (const *),
//  garantiamo la massima velocità di accesso alla memoria senza spreco di cicli di CPU,
//  un requisito essenziale per gestire flussi di dati tick-by-tick a bassissima latenza.
