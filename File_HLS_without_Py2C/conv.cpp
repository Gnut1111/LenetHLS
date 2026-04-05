#include "conv.h"

void Conv2D(fxp* Input_Conv,
            fxp* Output_Conv,
            fxp* conv_bias,
            fxp* conv_kernel,
            int h_in,
            int w_in,
            int c_in,
            int c_out,
            int k_size,
            int pool_stride,
            bool conv_layer,
            bool pool_layer,
            bool dense_layer,
            bool softmax,
            fxp &output_class
        )   
{
    // KHÃƒâ€?NG dÃƒÂ¹ng const Ã¡Â»Å¸ Ã„â€˜ÃƒÂ¢y

    int h_out = 1;
    int w_out = 1;
    int loop1 = 0, loop2 = 0, loop3 = 0, loop4 = 0, loop5 = 0, loop6 = 0;
    int base_h = 0;
    int base_w = 0;

    if (conv_layer) {
        h_out = h_in - k_size + 1;
        w_out = w_in - k_size + 1;

        loop1 = c_out;   // out channels
        loop2 = h_in - k_size + 1;   // out H
        loop3 = w_in - k_size + 1;   // out W
        loop4 = c_in;    // in channels
        loop5 = k_size;  // kernel H
        loop6 = k_size;  // kernel W
    }
    else if (pool_layer) {
        h_out = (h_in - k_size) / pool_stride + 1;
        w_out = (w_in - k_size) / pool_stride + 1;

        loop1 = c_in;    // mÃ¡Â»â€”i channel pool riÃƒÂªng
        loop2 = (h_in - k_size) / pool_stride + 1;
        loop3 = (w_in - k_size) / pool_stride + 1;
        loop4 = 1;       // khÃƒÂ´ng dÃƒÂ¹ng k thÃ¡ÂºÂ­t sÃ¡Â»Â±
        loop5 = k_size;
        loop6 = k_size;
    }
    else if (dense_layer) {
        loop1 = 1;
        loop2 = 1;
        loop3 = c_out;   // sÃ¡Â»â€˜ neuron output
        loop4 = c_in;    // sÃ¡Â»â€˜ input
        loop5 = 0;
        loop6 = 0;
    }

    fxp max_logit      = fxp(-100);
    int predicted_class = 0;
    fxp logits[10];         // giÃ¡ÂºÂ£ Ã„â€˜Ã¡Â»â€¹nh c_out <= 10
conv_oc:
    for (int n = 0; n < loop1; ++n)
    {
    conv_oh:
        for (int x = 0; x < loop2; ++x)
        {
        conv_ow:
            for (int y = 0; y < loop3; ++y)
            {
#pragma HLS PIPELINE II=1
                fxp_acc s = (pool_layer) ? (fxp_acc)-1e10 : (fxp_acc)0;

                base_h = (pool_layer) ? (x * pool_stride) : x;
                base_w = (pool_layer) ? (y * pool_stride) : y;

            conv_ic:
                for (int k = 0; k < loop4; ++k)
                {
#pragma HLS UNROLL factor = 2
                    // Dense: dot product Ã„â€˜Ã†Â¡n giÃ¡ÂºÂ£n
                    if (dense_layer) {
                        s += (fxp_acc)Input_Conv[k] *
                             (fxp_acc)conv_kernel[k * c_out + y];
                    }

                conv_ky:
                    for (int i = 0; i < loop5; ++i)
                    {
                    conv_kx:
                        for (int j = 0; j < loop6; ++j)
                        {
#pragma HLS PIPELINE II=1
                            int in_h = base_h + i;
                            int in_w = base_w + j;
                            int temp = (pool_layer) ? n : k;
                            int in_idx = (temp * h_in + in_h) * w_in + in_w;

                            if (conv_layer) {   
                                int ker_idx = ((n * loop4 + k) * loop5 + i) * loop6 + j;
                                s += (fxp_acc)conv_kernel[ker_idx] *
                                     (fxp_acc)Input_Conv[in_idx];
                            }
                            else if (pool_layer) {
                            	fxp_acc val = (fxp_acc) Input_Conv[in_idx];
                                if (val > s)
                                    s = val;
                            }
                        }
                    }
                }

                int temp1 = (conv_layer) ? n : y;
                if (!pool_layer) {
                    s += (fxp_acc)conv_bias[temp1];
                }

                fxp result = (s < (fxp_acc)0) ? (fxp)0 : (fxp)s;

                int temp2 = (dense_layer)
                            ? y
                            : (n * h_out * w_out + x * w_out + y);

                Output_Conv[temp2] = result;

                if (softmax) {
                    // thÃ†Â°Ã¡Â»?ng dÃƒÂ¹ng khi dense_layer + softmax
                    logits[y] = (fxp)s;
                }
            }
        }
    }

    if (softmax) {
    loop_argmax:
        for (int i = 0; i < c_out; i++)
        {
            if (logits[i] > max_logit)
            {
                max_logit       = logits[i];
                predicted_class = i;
            }
        }

        output_class = fxp(predicted_class);
    }
}
void Flatten(fxp input_Flatten[576],
             fxp output_Flatten[128],
             int h,
             int w,
             int c)
{
#pragma HLS INLINE
    int hs = 0;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            for (int k = 0; k < c; ++k)
            {
#pragma HLS PIPELINE II=1
                int in_idx = k * h * w + i * w + j;
                output_Flatten[hs++] = input_Flatten[in_idx];
            }
        }
    }
}