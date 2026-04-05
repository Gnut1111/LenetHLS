#ifndef CONV_H
#define CONV_H

#include <ap_fixed.h>

// Ä�á»‹nh nghÄ©a kiá»ƒu dá»¯ liá»‡u Fixed-point tÃ¹y theo yÃªu cáº§u project cá»§a báº¡n
// VÃ­ dá»¥: ap_fixed<Ä‘á»™_rá»™ng_bit, sá»‘_bit_nguyÃªn>
// á»ž Ä‘Ã¢y tÃ´i giáº£ Ä‘á»‹nh dÃ¹ng 16-bit vá»›i 8-bit nguyÃªn. Báº¡n cÃ³ thá»ƒ Ä‘iá»�u chá»‰nh láº¡i.
typedef ap_fixed<32, 16> fxp;
typedef ap_fixed<48, 24> fxp_acc; // Kiá»ƒu tÃ­ch lÅ©y Ä‘á»ƒ trÃ¡nh trÃ n sá»‘ khi nhÃ¢n cá»™ng

void Conv2D(
    fxp* Input_Conv,
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
);

void Flatten(
    fxp input_Flatten[576],
    fxp output_Flatten[128],
    int h,
    int w,
    int c
);

#endif // CONV_H
