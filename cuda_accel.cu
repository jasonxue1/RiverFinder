#include "cuda_accel.h"

#include <cuda_runtime.h>

#include "cubiomes/biomes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kMaxClimateOctaves = 16;
constexpr int kMaxSplineNodes = 42 + 151;

struct GpuClimateNoise {
    double amplitude = 0.0;
    int octcntA = 0;
    int octcntB = 0;
    PerlinNoise octA[kMaxClimateOctaves]{};
    PerlinNoise octB[kMaxClimateOctaves]{};
};

struct GpuSplineNode {
    int len = 0;
    int typ = 0;
    int isFix = 0;
    float fixValue = 0.0f;
    float loc[12]{};
    float der[12]{};
    int child[12]{};
};

struct GpuBiomeNoiseParams {
    GpuClimateNoise climate[NP_MAX]{};
    GpuSplineNode spline[kMaxSplineNodes]{};
    int splineRoot = -1;
};

struct ThreadCudaBuffers {
    int deviceId = -1;
    int *dRaw = nullptr;
    int *dRowPrefix = nullptr;
    int *dAreas = nullptr;
    int *dDxOut = nullptr;
    int *dDxIn = nullptr;
    CudaRiverCandidate *dCandidates = nullptr;
    int *dCandidateCount = nullptr;
    int *dMaxArea = nullptr;
    size_t rawCapacity = 0;
    size_t rowPrefixCapacity = 0;
    size_t areaCapacity = 0;
    size_t dxOutCapacity = 0;
    size_t dxInCapacity = 0;
    size_t candidateCapacity = 0;
    size_t candidateCountCapacity = 0;
    size_t maxAreaCapacity = 0;
    cudaStream_t stream = nullptr;
    GpuBiomeNoiseParams *dBiomeParams = nullptr;
    const BiomeNoise *cachedBiomeSource = nullptr;
    int cachedLutROut = -1;
    int cachedLutRIn = -1;
};

struct ThreadCudaContext {
    std::vector<ThreadCudaBuffers> perDevice;

    ~ThreadCudaContext()
    {
        for (ThreadCudaBuffers &buffers: perDevice)
        {
            if (buffers.deviceId < 0)
            {
                continue;
            }
            cudaSetDevice(buffers.deviceId);
            if (buffers.dAreas != nullptr)
            {
                cudaFree(buffers.dAreas);
            }
            if (buffers.dRowPrefix != nullptr)
            {
                cudaFree(buffers.dRowPrefix);
            }
            if (buffers.dDxOut != nullptr)
            {
                cudaFree(buffers.dDxOut);
            }
            if (buffers.dDxIn != nullptr)
            {
                cudaFree(buffers.dDxIn);
            }
            if (buffers.dCandidates != nullptr)
            {
                cudaFree(buffers.dCandidates);
            }
            if (buffers.dCandidateCount != nullptr)
            {
                cudaFree(buffers.dCandidateCount);
            }
            if (buffers.dMaxArea != nullptr)
            {
                cudaFree(buffers.dMaxArea);
            }
            if (buffers.dRaw != nullptr)
            {
                cudaFree(buffers.dRaw);
            }
            if (buffers.stream != nullptr)
            {
                cudaStreamDestroy(buffers.stream);
            }
            if (buffers.dBiomeParams != nullptr)
            {
                cudaFree(buffers.dBiomeParams);
            }
        }
    }
};

__global__ void rowPrefixKernel(
    const int *raw,
    int *rowPrefix,
    int W,
    int H
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H)
    {
        return;
    }

    const int rawBase = row * W;
    const int prefixBase = row * (W + 1);
    rowPrefix[prefixBase] = 0;
    int sum = 0;
    for (int x = 0; x < W; ++x)
    {
        sum += raw[rawBase + x];
        rowPrefix[prefixBase + x + 1] = sum;
    }
}

__global__ void ringAreaKernelFromRowPrefix(
    const int *rowPrefix,
    int *areas,
    int W,
    int ROut,
    int xCount,
    int zCount,
    int scale,
    const int *dxOut,
    const int *dxIn
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = xCount * zCount;
    if (idx >= total)
    {
        return;
    }

    const int localZ = idx / xCount;
    const int localX = idx - localZ * xCount;

    const int cx = localX + ROut;
    const int cz = localZ + ROut;

    int area = 0;
    for (int lut = 0; lut <= 2 * ROut; ++lut)
    {
        const int dz = lut - ROut;
        const int row = cz + dz;
        const int out = dxOut[lut];
        const int in = dxIn[lut];

        const int prefixBase = row * (W + 1);
        if (in < 0)
        {
            area += rowPrefix[prefixBase + cx + out + 1] - rowPrefix[prefixBase + cx - out];
        } else
        {
            area += rowPrefix[prefixBase + cx - in] - rowPrefix[prefixBase + cx - out];
            area += rowPrefix[prefixBase + cx + out + 1] - rowPrefix[prefixBase + cx + in + 1];
        }
    }

    areas[idx] = area * scale * scale;
}

__global__ void maxAreaKernel(const int *areas, int total, int *outMax)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    atomicMax(outMax, areas[idx]);
}

__global__ void filterCandidatesKernel(
    const int *areas,
    int xCount,
    int zCount,
    int ROut,
    int startX,
    int startZ,
    int scale,
    int minArea,
    double threshold,
    CudaRiverCandidate *outCandidates,
    int *outCount
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = xCount * zCount;
    if (idx >= total)
    {
        return;
    }

    const int worldArea = areas[idx];
    if (worldArea < minArea || static_cast<double>(worldArea) < threshold)
    {
        return;
    }

    const int localZ = idx / xCount;
    const int localX = idx - localZ * xCount;
    const int cx = localX + ROut;
    const int cz = localZ + ROut;
    const int outIdx = atomicAdd(outCount, 1);
    outCandidates[outIdx] = {
        startX + cx * scale,
        startZ + cz * scale,
        worldArea
    };
}

bool checkCuda(cudaError_t err, const char *stage)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << stage << ": " << cudaGetErrorString(err) << "\n";
        // Drain sticky error state so subsequent calls can retry instead of cascading failures.
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaGetLastError();
        return false;
    }
    return true;
}

bool ensureStream(ThreadCudaBuffers &buffers)
{
    if (buffers.stream != nullptr)
    {
        return true;
    }
    return checkCuda(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

ThreadCudaBuffers &getThreadBuffers(ThreadCudaContext &ctx, int deviceId)
{
    for (ThreadCudaBuffers &buffers: ctx.perDevice)
    {
        if (buffers.deviceId == deviceId)
        {
            return buffers;
        }
    }
    ctx.perDevice.push_back(ThreadCudaBuffers{});
    ThreadCudaBuffers &buffers = ctx.perDevice.back();
    buffers.deviceId = deviceId;
    return buffers;
}

template <typename T>
bool ensureCapacity(T *&ptr, size_t &capacity, size_t required, const char *stage)
{
    if (required <= capacity)
    {
        return true;
    }

    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
        capacity = 0;
    }

    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(T) * required), stage))
    {
        return false;
    }
    capacity = required;
    return true;
}

__device__ __forceinline__ double gpuLerp(double t, double a, double b)
{
    return a + t * (b - a);
}

__device__ __forceinline__ double gpuIndexedLerp(uint8_t idx, double a, double b, double c)
{
    switch (idx & 0xf)
    {
    case 0: return a + b;
    case 1: return -a + b;
    case 2: return a - b;
    case 3: return -a - b;
    case 4: return a + c;
    case 5: return -a + c;
    case 6: return a - c;
    case 7: return -a - c;
    case 8: return b + c;
    case 9: return -b + c;
    case 10: return b - c;
    case 11: return -b - c;
    case 12: return a + b;
    case 13: return -b + c;
    case 14: return -a + b;
    case 15: return -b - c;
    default: return 0;
    }
}

__device__ double gpuSamplePerlin(const PerlinNoise *noise, double d1, double d2, double d3)
{
    uint8_t h1, h2, h3;
    double t1, t2, t3;

    if (d2 == 0.0)
    {
        d2 = noise->d2;
        h2 = noise->h2;
        t2 = noise->t2;
    }
    else
    {
        d2 += noise->b;
        const double i2 = floor(d2);
        d2 -= i2;
        const int hi2 = static_cast<int>(i2);
        h2 = static_cast<uint8_t>(hi2);
        t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
    }

    d1 += noise->a;
    d3 += noise->c;

    const double i1 = floor(d1);
    const double i3 = floor(d3);
    d1 -= i1;
    d3 -= i3;

    const int hi1 = static_cast<int>(i1);
    const int hi3 = static_cast<int>(i3);
    h1 = static_cast<uint8_t>(hi1);
    h3 = static_cast<uint8_t>(hi3);

    t1 = d1 * d1 * d1 * (d1 * (d1 * 6.0 - 15.0) + 10.0);
    t3 = d3 * d3 * d3 * (d3 * (d3 * 6.0 - 15.0) + 10.0);

    const uint8_t *idx = noise->d;
    const uint8_t a1 = idx[h1] + h2;
    const uint8_t b1 = idx[h1 + 1] + h2;

    const uint8_t a2 = idx[a1] + h3;
    const uint8_t b2 = idx[b1] + h3;
    const uint8_t a3 = idx[a1 + 1] + h3;
    const uint8_t b3 = idx[b1 + 1] + h3;

    double l1 = gpuIndexedLerp(idx[a2], d1, d2, d3);
    double l2 = gpuIndexedLerp(idx[b2], d1 - 1, d2, d3);
    double l3 = gpuIndexedLerp(idx[a3], d1, d2 - 1, d3);
    double l4 = gpuIndexedLerp(idx[b3], d1 - 1, d2 - 1, d3);
    double l5 = gpuIndexedLerp(idx[a2 + 1], d1, d2, d3 - 1);
    double l6 = gpuIndexedLerp(idx[b2 + 1], d1 - 1, d2, d3 - 1);
    double l7 = gpuIndexedLerp(idx[a3 + 1], d1, d2 - 1, d3 - 1);
    double l8 = gpuIndexedLerp(idx[b3 + 1], d1 - 1, d2 - 1, d3 - 1);

    l1 = gpuLerp(t1, l1, l2);
    l3 = gpuLerp(t1, l3, l4);
    l5 = gpuLerp(t1, l5, l6);
    l7 = gpuLerp(t1, l7, l8);

    l1 = gpuLerp(t2, l1, l3);
    l5 = gpuLerp(t2, l5, l7);

    return gpuLerp(t3, l1, l5);
}

__device__ double gpuSampleOctave(const PerlinNoise *octaves, int octcnt, double x, double y, double z)
{
    double v = 0;
    for (int i = 0; i < octcnt; ++i)
    {
        const PerlinNoise *p = octaves + i;
        const double lf = p->lacunarity;
        const double ax = x * lf;
        const double ay = y * lf;
        const double az = z * lf;
        const double pv = gpuSamplePerlin(p, ax, ay, az);
        v += p->amplitude * pv;
    }
    return v;
}

__device__ double gpuSampleDoublePerlin(const GpuClimateNoise *noise, double x, double y, double z)
{
    const double f = 337.0 / 331.0;
    double v = 0;
    v += gpuSampleOctave(noise->octA, noise->octcntA, x, y, z);
    v += gpuSampleOctave(noise->octB, noise->octcntB, x * f, y * f, z * f);
    return v * noise->amplitude;
}

__device__ float gpuEvalSpline(const GpuBiomeNoiseParams *bn, int rootIdx, const float *vals)
{
    struct Frame {
        int idx;
        int mode;
        int left;
        int right;
        int i;
        float f;
    };

    Frame stack[64];
    int sp = 0;

    float memo[kMaxSplineNodes];
    int ready[kMaxSplineNodes];
    for (int i = 0; i < kMaxSplineNodes; ++i)
    {
        ready[i] = 0;
        memo[i] = 0.0f;
    }

    stack[sp++] = {rootIdx, 0, -1, -1, 0, 0.0f};

    while (sp > 0)
    {
        Frame &fr = stack[sp - 1];
        const GpuSplineNode &node = bn->spline[fr.idx];

        if (fr.mode == 0)
        {
            if (node.isFix != 0 || node.len == 1)
            {
                memo[fr.idx] = node.fixValue;
                ready[fr.idx] = 1;
                --sp;
                continue;
            }

            fr.f = vals[node.typ];
            fr.i = 0;
            for (; fr.i < node.len; ++fr.i)
            {
                if (node.loc[fr.i] >= fr.f)
                {
                    break;
                }
            }

            if (fr.i == 0 || fr.i == node.len)
            {
                if (fr.i)
                {
                    --fr.i;
                }
                fr.left = node.child[fr.i];
                fr.mode = 1;
                if (fr.left >= 0 && !ready[fr.left])
                {
                    stack[sp++] = {fr.left, 0, -1, -1, 0, 0.0f};
                }
                continue;
            }

            fr.left = node.child[fr.i - 1];
            fr.right = node.child[fr.i];
            fr.mode = 2;
            if (fr.left >= 0 && !ready[fr.left])
            {
                stack[sp++] = {fr.left, 0, -1, -1, 0, 0.0f};
            }
            continue;
        }

        if (fr.mode == 1)
        {
            if (fr.left >= 0 && !ready[fr.left])
            {
                stack[sp++] = {fr.left, 0, -1, -1, 0, 0.0f};
                continue;
            }
            const float v = memo[fr.left];
            memo[fr.idx] = v + node.der[fr.i] * (fr.f - node.loc[fr.i]);
            ready[fr.idx] = 1;
            --sp;
            continue;
        }

        if (fr.left >= 0 && !ready[fr.left])
        {
            stack[sp++] = {fr.left, 0, -1, -1, 0, 0.0f};
            continue;
        }
        if (fr.right >= 0 && !ready[fr.right])
        {
            stack[sp++] = {fr.right, 0, -1, -1, 0, 0.0f};
            continue;
        }

        const float g = node.loc[fr.i - 1];
        const float h = node.loc[fr.i];
        const float k = (fr.f - g) / (h - g);
        const float l = node.der[fr.i - 1];
        const float m = node.der[fr.i];
        const float n = memo[fr.left];
        const float o = memo[fr.right];
        const float p = l * (h - g) - (o - n);
        const float q = -m * (h - g) + (o - n);
        memo[fr.idx] = static_cast<float>(gpuLerp(k, n, o) + k * (1.0F - k) * gpuLerp(k, p, q));
        ready[fr.idx] = 1;
        --sp;
    }

    return memo[rootIdx];
}

__device__ int gpuSampleBiomeNoiseOnRiver(const GpuBiomeNoiseParams *bn, int x, int y, int z)
{
    // Match CPU call site: sampleBiomeNoiseOnRiver(..., SAMPLE_NO_SHIFT)
    const double px = x;
    const double pz = z;

    const float w = static_cast<float>(gpuSampleDoublePerlin(&bn->climate[NP_WEIRDNESS], px, 0, pz));
    if (w > 0.05f || w < -0.05f) return 0;

    const float c = static_cast<float>(gpuSampleDoublePerlin(&bn->climate[NP_CONTINENTALNESS], px, 0, pz));
    if (c < -0.19f) return 0;

    const float e = static_cast<float>(gpuSampleDoublePerlin(&bn->climate[NP_EROSION], px, 0, pz));
    if (c > 0.03f && e < -0.375f) return 0;

    const float t = static_cast<float>(gpuSampleDoublePerlin(&bn->climate[NP_TEMPERATURE], px, 0, pz));
    if (t < -0.45f) return 0;
    if (e > 0.55f && c >= -0.11f) return 0;

    const float npParam[] = {
        c,
        e,
        -3.0F * (fabsf(fabsf(w) - 0.6666667F) - 0.33333334F),
        w,
    };
    const double off = gpuEvalSpline(bn, bn->splineRoot, npParam) + 0.015F;
    const float d = static_cast<float>(1.0 - (y * 4) / 128.0 - 83.0 / 160.0 + off);
    if (d > 0 && d < 1)
    {
        if (0.8f < c && c < 1.0f) return 0;
    }
    return river;
}

__global__ void sampleRiverMaskKernel(
    const GpuBiomeNoiseParams *bn,
    int *raw,
    int W,
    int H,
    int startX,
    int startZ,
    int y,
    int scale
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H;
    if (idx >= total)
    {
        return;
    }

    const int z = idx / W;
    const int x = idx - z * W;
    const int worldX = startX + x * scale + scale / 2;
    const int worldZ = startZ + z * scale + scale / 2;
    const int id = gpuSampleBiomeNoiseOnRiver(bn, worldX / 4, y / 4 + 1, worldZ / 4);
    raw[idx] = (id == river) ? 1 : 0;
}

bool ensureDeviceBiomeParams(ThreadCudaBuffers &buffers, const BiomeNoise *bn)
{
    if (buffers.cachedBiomeSource == bn && buffers.dBiomeParams != nullptr)
    {
        return true;
    }

    if (buffers.dBiomeParams == nullptr)
    {
        if (!checkCuda(cudaMalloc(&buffers.dBiomeParams, sizeof(GpuBiomeNoiseParams)), "cudaMalloc(dBiomeParams)"))
        {
            return false;
        }
    }

    GpuBiomeNoiseParams params{};
    for (int i = 0; i < NP_MAX; ++i)
    {
        const DoublePerlinNoise &srcClimate = bn->climate[i];
        GpuClimateNoise &dstClimate = params.climate[i];
        dstClimate.amplitude = srcClimate.amplitude;

        const int octcntA = srcClimate.octA.octcnt;
        const int octcntB = srcClimate.octB.octcnt;
        if (octcntA < 0 || octcntB < 0 ||
            octcntA > kMaxClimateOctaves || octcntB > kMaxClimateOctaves ||
            srcClimate.octA.octaves == nullptr || srcClimate.octB.octaves == nullptr)
        {
            std::cerr << "CUDA error at ensureDeviceBiomeParams: invalid climate octave layout\n";
            return false;
        }

        dstClimate.octcntA = octcntA;
        dstClimate.octcntB = octcntB;
        for (int j = 0; j < octcntA; ++j)
        {
            dstClimate.octA[j] = srcClimate.octA.octaves[j];
        }
        for (int j = 0; j < octcntB; ++j)
        {
            dstClimate.octB[j] = srcClimate.octB.octaves[j];
        }
    }

    const uintptr_t stackBegin = reinterpret_cast<uintptr_t>(&bn->ss.stack[0]);
    const uintptr_t stackEnd = reinterpret_cast<uintptr_t>(&bn->ss.stack[42]);
    const uintptr_t fBegin = reinterpret_cast<uintptr_t>(&bn->ss.fstack[0]);
    const uintptr_t fEnd = reinterpret_cast<uintptr_t>(&bn->ss.fstack[151]);

    auto mapSplinePtr = [&](const Spline *ptr) -> int {
        if (ptr == nullptr)
        {
            return -1;
        }
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (p >= stackBegin && p < stackEnd)
        {
            const size_t idx = (p - stackBegin) / sizeof(Spline);
            if (idx < 42) return static_cast<int>(idx);
            return -1;
        }
        if (p >= fBegin && p < fEnd)
        {
            const size_t idx = (p - fBegin) / sizeof(FixSpline);
            if (idx < 151) return 42 + static_cast<int>(idx);
            return -1;
        }
        return -1;
    };

    params.splineRoot = mapSplinePtr(bn->sp);
    if (params.splineRoot < 0)
    {
        std::cerr << "CUDA error at ensureDeviceBiomeParams: invalid spline root\n";
        return false;
    }

    for (int i = 0; i < 42; ++i)
    {
        GpuSplineNode &dst = params.spline[i];
        const Spline &src = bn->ss.stack[i];
        dst.len = src.len;
        dst.typ = src.typ;
        dst.isFix = 0;
        dst.fixValue = 0.0f;
        for (int j = 0; j < 12; ++j)
        {
            dst.loc[j] = src.loc[j];
            dst.der[j] = src.der[j];
            dst.child[j] = -1;
        }
        for (int j = 0; j < src.len && j < 12; ++j)
        {
            dst.child[j] = mapSplinePtr(src.val[j]);
            if (dst.child[j] < 0)
            {
                std::cerr << "CUDA error at ensureDeviceBiomeParams: invalid spline child pointer\n";
                return false;
            }
        }
    }

    for (int i = 0; i < 151; ++i)
    {
        GpuSplineNode &dst = params.spline[42 + i];
        const FixSpline &src = bn->ss.fstack[i];
        dst.len = src.len;
        dst.typ = 0;
        dst.isFix = 1;
        dst.fixValue = src.val;
        for (int j = 0; j < 12; ++j)
        {
            dst.loc[j] = 0.0f;
            dst.der[j] = 0.0f;
            dst.child[j] = -1;
        }
    }

    if (!checkCuda(
        cudaMemcpyAsync(buffers.dBiomeParams, &params, sizeof(GpuBiomeNoiseParams), cudaMemcpyHostToDevice, buffers.stream),
        "cudaMemcpyAsync dBiomeParams H2D"))
    {
        buffers.cachedBiomeSource = nullptr;
        return false;
    }
    buffers.cachedBiomeSource = bn;
    return true;
}

bool ensureRingLut(ThreadCudaBuffers &buffers, int ROut, int RIn)
{
    if (buffers.cachedLutROut == ROut &&
        buffers.cachedLutRIn == RIn &&
        buffers.dDxOut != nullptr &&
        buffers.dDxIn != nullptr)
    {
        return true;
    }

    const int lutSize = 2 * ROut + 1;
    if (!ensureCapacity(buffers.dDxOut, buffers.dxOutCapacity, static_cast<size_t>(lutSize), "cudaMalloc(dDxOut)"))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dDxIn, buffers.dxInCapacity, static_cast<size_t>(lutSize), "cudaMalloc(dDxIn)"))
    {
        return false;
    }

    std::vector<int> dxOut(lutSize);
    std::vector<int> dxIn(lutSize, -1);
    for (int dz = -ROut; dz <= ROut; ++dz)
    {
        const int i = dz + ROut;
        dxOut[i] = static_cast<int>(floor(std::sqrt(static_cast<double>(ROut * ROut - dz * dz))));
        if (std::abs(dz) <= RIn)
        {
            dxIn[i] = static_cast<int>(floor(std::sqrt(static_cast<double>(RIn * RIn - dz * dz))));
        }
    }

    if (!checkCuda(
        cudaMemcpyAsync(buffers.dDxOut, dxOut.data(), sizeof(int) * lutSize, cudaMemcpyHostToDevice, buffers.stream),
        "cudaMemcpyAsync dDxOut H2D"))
    {
        buffers.cachedLutROut = -1;
        buffers.cachedLutRIn = -1;
        return false;
    }
    if (!checkCuda(
        cudaMemcpyAsync(buffers.dDxIn, dxIn.data(), sizeof(int) * lutSize, cudaMemcpyHostToDevice, buffers.stream),
        "cudaMemcpyAsync dDxIn H2D"))
    {
        buffers.cachedLutROut = -1;
        buffers.cachedLutRIn = -1;
        return false;
    }
    buffers.cachedLutROut = ROut;
    buffers.cachedLutRIn = RIn;
    return true;
}

bool runRingPipeline(
    ThreadCudaBuffers &buffers,
    int W,
    int H,
    int startX,
    int startZ,
    int scale,
    int minArea,
    double f,
    std::vector<CudaRiverCandidate> &outCandidates,
    int &outMaxArea
)
{
    outCandidates.clear();
    outMaxArea = 0;

    const int ROut = 128 / scale;
    const int RIn = 24 / scale;
    const int xCount = W - 2 * ROut;
    const int zCount = H - 2 * ROut;
    if (xCount <= 0 || zCount <= 0)
    {
        return true;
    }

    const int areaSize = xCount * zCount;
    const int rowPrefixSize = H * (W + 1);
    if (!ensureCapacity(buffers.dRowPrefix, buffers.rowPrefixCapacity, static_cast<size_t>(rowPrefixSize), "cudaMalloc(dRowPrefix)"))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dAreas, buffers.areaCapacity, static_cast<size_t>(areaSize), "cudaMalloc(dAreas)"))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dCandidates, buffers.candidateCapacity, static_cast<size_t>(areaSize), "cudaMalloc(dCandidates)"))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dCandidateCount, buffers.candidateCountCapacity, static_cast<size_t>(1), "cudaMalloc(dCandidateCount)"))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dMaxArea, buffers.maxAreaCapacity, static_cast<size_t>(1), "cudaMalloc(dMaxArea)"))
    {
        return false;
    }
    if (!ensureRingLut(buffers, ROut, RIn))
    {
        return false;
    }

    const int threads = 256;
    const int rowBlocks = (H + threads - 1) / threads;
    rowPrefixKernel<<<rowBlocks, threads, 0, buffers.stream>>>(buffers.dRaw, buffers.dRowPrefix, W, H);
    if (!checkCuda(cudaGetLastError(), "rowPrefixKernel launch"))
    {
        return false;
    }

    const int areaBlocks = (areaSize + threads - 1) / threads;
    ringAreaKernelFromRowPrefix<<<areaBlocks, threads, 0, buffers.stream>>>(
        buffers.dRowPrefix,
        buffers.dAreas,
        W,
        ROut,
        xCount,
        zCount,
        scale,
        buffers.dDxOut,
        buffers.dDxIn
    );
    if (!checkCuda(cudaGetLastError(), "ringAreaKernelFromRowPrefix launch"))
    {
        return false;
    }

    if (!checkCuda(cudaMemsetAsync(buffers.dMaxArea, 0, sizeof(int), buffers.stream), "cudaMemsetAsync dMaxArea"))
    {
        return false;
    }
    maxAreaKernel<<<areaBlocks, threads, 0, buffers.stream>>>(buffers.dAreas, areaSize, buffers.dMaxArea);
    if (!checkCuda(cudaGetLastError(), "maxAreaKernel launch"))
    {
        return false;
    }

    if (!checkCuda(
        cudaMemcpyAsync(&outMaxArea, buffers.dMaxArea, sizeof(int), cudaMemcpyDeviceToHost, buffers.stream),
        "cudaMemcpyAsync maxArea D2H"))
    {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(buffers.stream), "cudaStreamSynchronize after max"))
    {
        return false;
    }
    if (outMaxArea <= 0)
    {
        return true;
    }

    const double threshold = static_cast<double>(outMaxArea) * f;
    if (!checkCuda(cudaMemsetAsync(buffers.dCandidateCount, 0, sizeof(int), buffers.stream), "cudaMemsetAsync dCandidateCount"))
    {
        return false;
    }
    filterCandidatesKernel<<<areaBlocks, threads, 0, buffers.stream>>>(
        buffers.dAreas,
        xCount,
        zCount,
        ROut,
        startX,
        startZ,
        scale,
        minArea,
        threshold,
        buffers.dCandidates,
        buffers.dCandidateCount
    );
    if (!checkCuda(cudaGetLastError(), "filterCandidatesKernel launch"))
    {
        return false;
    }

    int candidateCount = 0;
    if (!checkCuda(
        cudaMemcpyAsync(&candidateCount, buffers.dCandidateCount, sizeof(int), cudaMemcpyDeviceToHost, buffers.stream),
        "cudaMemcpyAsync candidateCount D2H"))
    {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(buffers.stream), "cudaStreamSynchronize after filter"))
    {
        return false;
    }
    if (candidateCount <= 0)
    {
        return true;
    }

    outCandidates.resize(static_cast<size_t>(candidateCount));
    if (!checkCuda(
        cudaMemcpyAsync(
            outCandidates.data(),
            buffers.dCandidates,
            sizeof(CudaRiverCandidate) * static_cast<size_t>(candidateCount),
            cudaMemcpyDeviceToHost,
            buffers.stream),
        "cudaMemcpyAsync candidates D2H"))
    {
        outCandidates.clear();
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(buffers.stream), "cudaStreamSynchronize after candidates copy"))
    {
        outCandidates.clear();
        return false;
    }

    return true;
}

} // namespace

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
    int deviceId
)
{
    outCandidates.clear();
    outMaxArea = 0;

    if (!checkCuda(cudaSetDevice(deviceId), "cudaSetDevice"))
    {
        return false;
    }

    const int rawSize = W * H;

    static thread_local ThreadCudaContext threadContext;
    ThreadCudaBuffers &buffers = getThreadBuffers(threadContext, deviceId);
    if (!ensureStream(buffers))
    {
        return false;
    }

    if (!ensureCapacity(buffers.dRaw, buffers.rawCapacity, static_cast<size_t>(rawSize), "cudaMalloc(dRaw)"))
    {
        return false;
    }
    bool ok = checkCuda(
        cudaMemcpyAsync(buffers.dRaw, raw, sizeof(int) * rawSize, cudaMemcpyHostToDevice, buffers.stream),
        "cudaMemcpyAsync raw H2D"
    );
    if (!ok)
    {
        return false;
    }
    return runRingPipeline(buffers, W, H, startX, startZ, scale, minArea, f, outCandidates, outMaxArea);
}

int cudaGetVisibleDeviceCount()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        return 0;
    }
    if (count < 0)
    {
        return 0;
    }
    return count;
}

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
    int deviceId
)
{
    outCandidates.clear();
    outMaxArea = 0;

    if (!checkCuda(cudaSetDevice(deviceId), "cudaSetDevice"))
    {
        return false;
    }

    const int rawSize = W * H;

    static thread_local ThreadCudaContext threadContext;
    ThreadCudaBuffers &buffers = getThreadBuffers(threadContext, deviceId);
    if (!ensureStream(buffers))
    {
        return false;
    }
    if (!ensureCapacity(buffers.dRaw, buffers.rawCapacity, static_cast<size_t>(rawSize), "cudaMalloc(dRaw)"))
    {
        return false;
    }
    if (!ensureDeviceBiomeParams(buffers, bn))
    {
        return false;
    }

    bool ok = true;
    const int threads = 256;
    const int sampleBlocks = (rawSize + threads - 1) / threads;
    sampleRiverMaskKernel<<<sampleBlocks, threads, 0, buffers.stream>>>(buffers.dBiomeParams, buffers.dRaw, W, H, startX, startZ, y, scale);
    ok = checkCuda(cudaGetLastError(), "sampleRiverMaskKernel launch");

    if (!ok)
    {
        return false;
    }
    return runRingPipeline(buffers, W, H, startX, startZ, scale, minArea, f, outCandidates, outMaxArea);
}
