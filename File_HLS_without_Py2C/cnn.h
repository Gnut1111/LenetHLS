#ifndef CNN_H
#define CNN_H

#include <ap_fixed.h>

typedef ap_fixed<32,16>  fxp;

// InModel : ảnh MNIST 28x28 flatten -> 784 phần tử
// OutModel0 : output (logit hoặc class) 1 phần tử
// Weights : mảng chứa toàn bộ tr�?ng số + bias, kích thước 5738
void CNN(fxp InModel[784], fxp &OutModel0, fxp Weights[5738]);

#endif // CNN_H
