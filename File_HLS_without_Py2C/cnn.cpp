#include "conv.h"
#include "cnn.h"
#include <ap_axi_sdata.h>
#include <ap_fixed.h>

typedef ap_fixed<32,16> fxp;

void CNN(fxp InModel[784], fxp &OutModel0, fxp Weights[5738]) {
#pragma HLS INTERFACE m_axi     port=InModel  offset=slave bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi     port=Weights  offset=slave bundle=gmem1 depth=5738
#pragma HLS INTERFACE s_axilite port=OutModel0           bundle=control
#pragma HLS INTERFACE s_axilite port=return              bundle=control

    // Buffer trung gian vá»›i memory binding
    fxp conv1_out[2304];
    fxp pool1_out[576];
    fxp flatten[128];
    fxp dense0_out[32];
    fxp dense1_out[16];
    fxp dense2_out[10];

    fxp dummy_class = 0;

    // Conv1: 28x28x1 -> 24x24x4
    Conv2D(InModel, conv1_out, &Weights[100], &Weights[0],
           28, 28, 1, 4, 5, 1,
           true, false, false, false, dummy_class);

    // Pool1: 24x24x4 -> 12x12x4
    Conv2D(conv1_out, pool1_out, &Weights[0], &Weights[0],
           24, 24, 4, 4,  // â†? Sá»¬A: c_out = 4 thay vÃ¬ 0
           2, 2,
           false, true, false, false, dummy_class);

    // Conv2: 12x12x4 -> 8x8x8
    Conv2D(pool1_out, conv1_out, &Weights[904], &Weights[104],
           12, 12, 4, 8, 5, 1,
           true, false, false, false, dummy_class);

    // Pool2: 8x8x8 -> 4x4x8
    Conv2D(conv1_out, pool1_out, &Weights[0], &Weights[0],
           8, 8, 8, 8,  // â†? Sá»¬A: c_out = 8 thay vÃ¬ 0
           2, 2,
           false, true, false, false, dummy_class);

    int hs = 0;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
#pragma HLS PIPELINE II=1
                int in_idx = k * 4 * 4 + i * 4 + j;
                flatten[hs++] = pool1_out[in_idx];
            }
        }
    }

    // Dense layers
    Conv2D(flatten, dense0_out, &Weights[5008], &Weights[912],
           1, 1, 128, 32, 1, 1,
           false, false, true, false, dummy_class);

    Conv2D(dense0_out, dense1_out, &Weights[5552], &Weights[5040],
           1, 1, 32, 16, 1, 1,
           false, false, true, false, dummy_class);

    // Output layer with argmax
    Conv2D(dense1_out, dense2_out, &Weights[5728], &Weights[5568],
           1, 1, 16, 10, 1, 1,
           false, false, true, true, OutModel0);
}
