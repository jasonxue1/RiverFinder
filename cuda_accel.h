#pragma once

#include "cubiomes/biomenoise.h"

#include <vector>

struct CudaRiverCandidate {
    int x;
    int z;
    int area;
};

// Returns true when CUDA computation succeeded.
// Returns false when CUDA initialization/kernel execution fails and caller should fallback to CPU.
bool cudaRingSearchCandidates(
    const int *raw,
    int W,
    int H,
    int startX,
    int startZ,
    int scale,
    int minArea,
    double f,
    std::vector<CudaRiverCandidate> &outCandidates,
    int &outMaxArea,
    int deviceId = 0
);

// Returns visible CUDA device count, or 0 when query fails.
int cudaGetVisibleDeviceCount();

// GPU path for scale>1: sample river mask on GPU from BiomeNoise, then run ring search.
bool cudaRingSearchCandidatesFromBiomeNoise(
    const BiomeNoise *bn,
    int y,
    int W,
    int H,
    int startX,
    int startZ,
    int scale,
    int minArea,
    double f,
    std::vector<CudaRiverCandidate> &outCandidates,
    int &outMaxArea,
    int deviceId = 0
);
