# 🧠 Handwritten Digit Recognition on FPGA using HLS

An embedded AI project implementing the **LeNet-5 CNN architecture** on FPGA via **High-Level Synthesis (HLS)**, targeting the MNIST handwritten digit dataset.

> Achieves **98.7% accuracy** on hardware with significantly reduced resource usage compared to the state of the art.

---

## 📋 Overview

This project presents an end-to-end pipeline for deploying a Convolutional Neural Network on FPGA:

1. **Train** a compact LeNet-5 variant in Python (TensorFlow/Keras) on the MNIST dataset
2. **Convert** the trained model to C++ using the [Py2C](https://github.com/WangNe2207/Py2C_new) tool
3. **Synthesize** the C++ code into an IP Core using Vitis HLS with hardware optimization directives
4. **Integrate** the IP Core into a SoC on the Zynq UltraScale+ MPSoC platform via AXI interfaces
5. **Validate** on real hardware with 1,000 MNIST test images

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Zynq UltraScale+ MPSoC                     │
│                                                             │
│  ┌─────────────────┐        ┌────────────────────────────┐  │
│  │  Processing     │        │  Programmable Logic (PL)   │  │
│  │  System (PS)    │        │                            │  │
│  │                 │        │  ┌──────────────────────┐  │  │
│  │  - Load weights │ AXI4   │  │    cnn_top_0         │  │  │
│  │  - Send image   ├───────►│  │  (LeNet-5 IP Core)   │  │  │
│  │  - Read result  │ Lite   │  │                      │  │  │
│  │                 │        │  └──────────┬───────────┘  │  │
│  └────────┬────────┘        │             │ AXI4         │  │
│           │                 │  ┌──────────▼───────────┐  │  │
│           │    AXI4         │  │  BRAM Controller     │  │  │
│           └────────────────►│  │  inmodel | weights   │  │  │
│                             │  └──────────────────────┘  │  │
│                             └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 CNN Architecture

A compact, FPGA-optimized variant of LeNet-5, reducing parameters by **~10.7×** vs. the original while maintaining high accuracy.

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| Input | — | 28 × 28 × 1 | 0 |
| C1 | Conv2D (4 filters, 5×5) + ReLU | 24 × 24 × 4 | 104 |
| S2 | MaxPool (2×2) | 12 × 12 × 4 | 0 |
| C3 | Conv2D (8 filters, 5×5) + ReLU | 8 × 8 × 8 | 808 |
| S4 | MaxPool (2×2) | 4 × 4 × 8 | 0 |
| Flatten | — | 128 | 0 |
| Dropout (0.2) | — | 128 | 0 |
| F5 | Dense + ReLU | 32 | 4,128 |
| Dropout (0.3) | — | 32 | 0 |
| F6 | Dense + ReLU | 16 | 528 |
| Output | Dense + Softmax | 10 | 170 |
| **Total** | | | **5,738** |

---

## 🔧 Hardware & Tools

| Item | Details |
|---|---|
| **FPGA Platform** | Zynq UltraScale+ MPSoC |
| **HLS Tool** | Vitis HLS (Vivado) |
| **Integration** | Vivado Design Suite |
| **SW Development** | Vitis SDK (C bare-metal) |
| **Model Training** | Python, TensorFlow / Keras |
| **Model Conversion** | [Py2C](https://github.com/WangNe2207/Py2C_new) |
| **Dataset** | MNIST (60K train / 10K test) |
| **Data Format** | Fixed-point `ap_fixed<16, 8>` |

---

## ⚙️ HLS Optimization Techniques

| Directive | Applied To | Effect |
|---|---|---|
| `#pragma HLS PIPELINE` | Convolution loops | Reduces Initiation Interval (II), increases throughput |
| `#pragma HLS UNROLL` | Filter loops | Parallel DSP execution across filters |
| `#pragma HLS INLINE` | Helper functions | Flattens hierarchy, enables cross-boundary optimization |
| Function merging | All CNN layers | Shared buffers & logic, reduces BRAM and DSP |
| Fixed-point arithmetic | All computation | Replaces float, drastically cuts LUT/FF/DSP usage |

---

## 📊 Resource Utilization

### Before vs. After Optimization

| Resource | Before (Py2C default) | After (optimized) | Reduction |
|---|---|---|---|
| **BRAM** | 40 | **14** | ↓ 65.0% |
| **DSP** | 242 | **116** | ↓ 52.1% |
| **FF** | 20,097 | **7,195** | ↓ 64.2% |
| **LUT** | 25,004 | **14,478** | ↓ 42.1% |

### Comparison with Published Work (ICCSS 2019)

| Resource | This Project | Reference Paper |
|---|---|---|
| **BRAM** | **14** | 119.5 |
| **DSP** | **116** | 125 |
| **LUT** | **14,478** | 14,659 |
| **FF** | **7,195** | 14,172 |

> Reference: Rongshi & Yongming, *"Accelerator Implementation of LeNet-5 CNN Based on FPGA with HLS"*, ICCSS 2019.

---

## 🗺️ AXI Address Map

| Region | Base Address | Size | Purpose |
|---|---|---|---|
| `s_axi_control` | `0x00_A000_0000` | 64 KB | Control registers (`ap_start`, `ap_done`, `ap_idle`) |
| `s_axi_control_r` | `0x00_A001_0000` | 64 KB | Input parameter registers (BRAM base addresses) |
| `inmodel_bram_ctrl` | `0x00_A002_0000` | 8 KB | Input image buffer |
| `weights_bram_ctrl` | `0x00_A002_8000` | 32 KB | Model weights & biases |

The IP Core uses **two separate AXI4 Master ports** (`gmem0` and `gmem1`) to access the image buffer and weight buffer simultaneously, maximizing memory bandwidth.

---

## 🔄 Inference Flow (Software)

```
1. Initialize system & configure AXI peripherals
2. Load all weights → write to weights_bram (0xA002_8000)
3. For each test image:
   a. Write pixel data (Fixed-point) → inmodel_bram (0xA002_0000)
   b. Write BRAM base addresses to s_axi_control_r registers
   c. Write ap_start = 1 to s_axi_control
   d. Poll ap_done flag until hardware finishes (~3 ms)
   e. Read predicted digit from result register
   f. Compare with ground-truth label, accumulate accuracy
4. Print final accuracy
```

---

## 📈 Training Results

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|---|---|---|---|---|
| 1 | 0.512 | 84.2% | 0.312 | 90.8% |
| 5 | 0.112 | 96.5% | 0.098 | 97.1% |
| 10 | 0.078 | 97.8% | 0.065 | 98.0% |
| 20 | 0.055 | 98.4% | 0.048 | 98.3% |
| 40 | 0.042 | 98.7% | 0.041 | 98.4% |

**Final hardware accuracy: 98.7%** on 1,000 MNIST test images.

---

## 📁 Project Structure

```
project/
├── Python/
│   ├── train.py               # Model training (TensorFlow/Keras)
│   └── export_weights.py      # Export weights for Py2C
├── Py2C_new/
│   ├── CNN.cpp                # Auto-generated C++ from Py2C
│   ├── CNN.h
│   └── CNN_tb.cpp             # C-simulation testbench (50 MNIST images)
├── HLS/
│   ├── CNN_opt.cpp            # Manually optimized C++ with HLS pragmas
│   ├── CNN_opt.h
│   └── hls_config.tcl         # Vitis HLS project script
├── Vivado/
│   └── block_design/          # SoC block design (IP Core + AXI interconnect)
├── Vitis_SW/
│   ├── main.c                 # Bare-metal inference application
│   └── test_data.h            # 1,000 MNIST test images (Fixed-point format)
└── README.md
```

---

## 🚀 Getting Started

**Requirements:**
- Vitis / Vivado Design Suite
- Vitis SDK
- Python 3.x + TensorFlow 2.x
- [Py2C](https://github.com/WangNe2207/Py2C_new)

**Steps:**

**1. Train the model**
```bash
cd Python
python train.py    # outputs model.h5
```

**2. Convert to C++ with Py2C**

Follow the [Py2C documentation](https://github.com/WangNe2207/Py2C_new) to generate `CNN.cpp` and `CNN.h` from `model.h5`.

**3. Run C-Simulation in Vitis HLS**

Open your HLS project and ensure:
- `CNN_opt.cpp` is added as a **Source** file
- `CNN_tb.cpp` is added as a **Testbench** file

Then run `csim_design`. Expected accuracy: ~98%.

> ⚠️ Common error: `undefined reference to CNN(...)` — this means `CNN_opt.cpp` was not added as a Source file. See [Troubleshooting](#troubleshooting) below.

**4. Synthesize & Export IP**
```
Run csynth_design → Export RTL → Package as IP Core
```

**5. Integrate in Vivado**
- Import IP Core into block design
- Connect AXI interfaces per the address map above
- Generate bitstream and program the device

**6. Deploy on hardware**
- Build `main.c` in Vitis SDK
- Run on board — results print to UART console

---

## 🐛 Troubleshooting

**`undefined reference to CNN(...)`**

This is a linker error, not a compile error. The testbench calls `CNN()` but the linker cannot find its definition. Fix:
- In Vitis HLS, go to **Explorer → Source** and verify `CNN_opt.cpp` is listed there (not only under Testbench).
- In your TCL script, ensure you have both:
  ```tcl
  add_files CNN_opt.cpp        # design source
  add_files -tb CNN_tb.cpp     # testbench only
  ```
- Check that the function signature in `CNN_opt.h` exactly matches the definition in `CNN_opt.cpp` — pay attention to pointer (`*`) vs. reference (`&`) parameters and `ap_fixed` template arguments.

---

## 🔮 Future Work

- **INT8 Quantization:** 4× memory reduction, faster inference with minimal accuracy loss
- **DDR4 + DMA:** Replace on-chip BRAM with external DDR4 to support larger models (VGG16, ResNet, YOLO)
- **Dataflow & Systolic Array:** Apply HLS Dataflow pragma and systolic array patterns for maximum parallel throughput
- **Larger datasets:** Extend to EMNIST, CIFAR-10, or custom handwriting datasets

---

## 📚 References

1. LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition*. IEEE.
2. Rongshi & Yongming (2019). *Accelerator Implementation of LeNet-5 CNN Based on FPGA with HLS*. ICCSS 2019, pp. 64–67.
3. Solovyev et al. (2018). *Fixed-Point CNN for Real-Time Video Processing in FPGA*. arXiv:1808.09945.
4. Xilinx/AMD (2023). *AXI Basics 1 — Introduction to AXI*. Adaptive Support Documentation.
5. Wang, N. (2023). *Py2C_new*. GitHub: https://github.com/WangNe2207/Py2C_new
6. LeCun, Cortes & Burges (1998). *The MNIST Database of Handwritten Digits*.
7. Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.

---

*University of Information Technology, VNU-HCM — 2026*
