#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "conv.h"   // fxp type definition
#include "cnn.h"    // CNN() declaration

// ============================================================
//  Testbench for C-Simulation in Vitis HLS
//  Reads weights, input images, and labels from .txt files,
//  runs inference for each image, prints per-image result
//  and final accuracy.
// ============================================================

#define NUMBER_OF_PICTURES  50   // how many images to test
#define CHANNELS             1   // input channels (grayscale)
#define IMG_SIZE            (CHANNELS * 28 * 28)   // 784

int main()
{
    // ── 1. Load weights ──────────────────────────────────────
    // Weights array must be exactly 5738 elements.
    fxp Weights[5738];

    FILE* fWeights = fopen("Float_Weights.txt", "r");
    if (!fWeights) {
        printf("[ERROR] Cannot open Float_Weights.txt\n");
        return -1;
    }
    for (int i = 0; i < 5738; i++) {
        float tmp;
        fscanf(fWeights, "%f", &tmp);
        Weights[i] = (fxp)tmp;
    }
    fclose(fWeights);
    printf("[INFO] Weights loaded OK\n");

    // ── 2. Load input images ──────────────────────────────────
    // X.txt contains NUMBER_OF_PICTURES * 784 float values,
    // stored row by row (one image after another).
    float* InRaw = (float*)malloc(NUMBER_OF_PICTURES * IMG_SIZE * sizeof(float));
    if (!InRaw) { printf("[ERROR] malloc InRaw\n"); return -1; }

    FILE* fInput = fopen("X.txt", "r");
    if (!fInput) {
        printf("[ERROR] Cannot open X.txt\n");
        return -1;
    }
    for (int i = 0; i < NUMBER_OF_PICTURES * IMG_SIZE; i++) {
        fscanf(fInput, "%f", &InRaw[i]);
    }
    fclose(fInput);
    printf("[INFO] Images loaded OK\n");

    // ── 3. Load labels ────────────────────────────────────────
    int Labels[NUMBER_OF_PICTURES];
    FILE* fLabel = fopen("Y.txt", "r");
    if (!fLabel) {
        printf("[ERROR] Cannot open Y.txt\n");
        return -1;
    }
    for (int i = 0; i < NUMBER_OF_PICTURES; i++) {
        float tmp;
        fscanf(fLabel, "%f", &tmp);
        Labels[i] = (int)tmp;
    }
    fclose(fLabel);
    printf("[INFO] Labels loaded OK\n");

    // ── 4. Run inference ──────────────────────────────────────
    int correct = 0;

    for (int i = 0; i < NUMBER_OF_PICTURES; i++)
    {
        // Copy one image into a local fixed-size array
        // (CNN() expects fxp InModel[784])
        fxp Image[IMG_SIZE];
        int base = i * IMG_SIZE;
        for (int k = 0; k < IMG_SIZE; k++) {
            Image[k] = (fxp)InRaw[base + k];
        }

        // Run the CNN accelerator
        fxp OutModel0 = 0;
        CNN(Image, OutModel0, Weights);

        int predicted = (int)OutModel0;
        int label     = Labels[i];

        if (predicted == label) correct++;

        printf("[%3d] label=%d  predicted=%d  %s\n",
               i, label, predicted,
               (predicted == label) ? "OK" : "WRONG");
    }

    // ── 5. Print summary ──────────────────────────────────────
    float accuracy = (float)correct / (float)NUMBER_OF_PICTURES * 100.0f;
    printf("\n=== Accuracy: %d / %d = %.2f%% ===\n",
           correct, NUMBER_OF_PICTURES, accuracy);

    free(InRaw);
    return 0;
}
