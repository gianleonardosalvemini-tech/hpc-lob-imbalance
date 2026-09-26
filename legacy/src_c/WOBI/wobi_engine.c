#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LOB_DEPTH 10

typedef struct {
    double price;
    double volume;
} LOBLevel;

typedef struct {
    LOBLevel bids[LOB_DEPTH];
    LOBLevel asks[LOB_DEPTH];
} LOBState;

typedef struct {
    double weighted_micro_price;
    double imbalance_pressure;
} SignalMetrics;

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// Funzione richiamata da Python per estrarre la pressione e il fair value
EXPORT SignalMetrics compute_signals(const LOBState* state, double alpha) {
    SignalMetrics metrics;
    double weighted_bid_vol = 0.0, weighted_ask_vol = 0.0;
    double micro_num = 0.0, micro_den = 0.0;
    
    // Decadimento esponenziale calcolato una volta sola per efficienza
    double current_weight = 1.0;
    double decay_factor = exp(-alpha);

    for(int i = 0; i < LOB_DEPTH; i++) {
        double b_v = state->bids[i].volume;
        double a_v = state->asks[i].volume;
        double b_p = state->bids[i].price;
        double a_p = state->asks[i].price;

        // Calcolo della pressione (Imbalance)
        weighted_bid_vol += b_v * current_weight;
        weighted_ask_vol += a_v * current_weight;

        // Calcolo del Fair Value pesato (incrocio di volumi e prezzi)
        micro_num += current_weight * (b_v * a_p + a_v * b_p);
        micro_den += current_weight * (b_v + a_v);

        current_weight *= decay_factor;
    }

    double total_vol = weighted_bid_vol + weighted_ask_vol;
    metrics.imbalance_pressure = (total_vol > 0.0) ? ((weighted_bid_vol - weighted_ask_vol) / total_vol) : 0.0;
    metrics.weighted_micro_price = (micro_den > 0.0) ? (micro_num / micro_den) : ((state->bids[0].price + state->asks[0].price)/2.0);

    return metrics;
}
