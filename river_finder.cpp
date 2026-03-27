#include "cubiomes/generator.h"
#include "cubiomes/biomes.h"
#include <vector>
#include <algorithm>
#include  <queue>
#include <iostream>
#include <chrono>
#include "Thread.h"
#include <map>
#include <functional>
#include <string>
#include <optional>
#include <cctype>
#include <cstdlib>
#include <charconv>
#include <unordered_map>
#include <cstdint>
#include <sstream>
#include <mutex>
#include <deque>
#ifdef RIVERFINDER_ENABLE_CUDA
#include "cuda_accel.h"
#endif
#define AREA_SIZE 256// 搜索步长（不重叠）

/* ================== 河流判定 ================== */

template<typename Func>
void measure_time(Func func, const std::string &name = "Function")
{
    auto start = std::chrono::high_resolution_clock::now();

    // 执行传入的函数
    func();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << name << " took: "
            << duration.count() << " microseconds ("
            << duration.count() / 1000.0 << " ms)" << std::endl;
}

static std::string formatDurationMs(long long ms)
{
    if (ms < 0)
    {
        ms = 0;
    }
    const long long totalSeconds = ms / 1000;
    const long long h = totalSeconds / 3600;
    const long long m = (totalSeconds % 3600) / 60;
    const long long s = totalSeconds % 60;

    std::ostringstream oss;
    if (h > 0)
    {
        oss << h << "h " << m << "m " << s << "s";
    } else if (m > 0)
    {
        oss << m << "m " << s << "s";
    } else
    {
        oss << s << "s";
    }
    return oss.str();
}

static inline int isRiverBiome(int id)
{
    return (id == river) * 0 + (id == dripstone_caves) * 10;
}

static inline int isDripstone_caves(int id)
{
    return (id == dripstone_caves);
}

static inline int value(int id)
{
    return (id == river);
}
struct Point {
    int x = 0;
    int y = 0;

    std::strong_ordering operator<=>(const Point &) const = default;
};



struct Res {
    Point point;
    int area = 0;

    bool operator<(const Res &other) const noexcept
    {
        // 首先按面积降序，面积相同时按坐标排序
        if (area != other.area) return area < other.area;
        return true;
    }

    Res() = default;

    Res(Point point, int area) : point(point), area(area)
    {
    };
};

enum class ComputeMode {
    Auto,
    Cpu,
    Cuda
};

static std::string toLower(std::string value)
{
    for (char &c: value)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

static std::optional<ComputeMode> parseComputeMode(const std::string &raw)
{
    const std::string mode = toLower(raw);
    if (mode == "auto")
    {
        return ComputeMode::Auto;
    }
    if (mode == "cpu")
    {
        return ComputeMode::Cpu;
    }
    if (mode == "cuda")
    {
        return ComputeMode::Cuda;
    }
    return std::nullopt;
}

template<typename T>
static bool parseNumberArg(const char *text, T &out)
{
    if (text == nullptr)
    {
        return false;
    }
    const char *begin = text;
    const char *end = text + std::char_traits<char>::length(text);
    auto result = std::from_chars(begin, end, out);
    return result.ec == std::errc{} && result.ptr == end;
}

static bool isCudaModeBuilt()
{
#ifdef RIVERFINDER_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

static ComputeMode resolveComputeMode(ComputeMode selectedMode)
{
    const bool cudaBuilt = isCudaModeBuilt();
    if (selectedMode == ComputeMode::Auto)
    {
        if (cudaBuilt)
        {
            std::cout << "mode=auto -> using cuda backend\n";
            return ComputeMode::Cuda;
        }
        std::cout << "mode=auto -> using cpu backend (cuda backend not built)\n";
        return ComputeMode::Cpu;
    }

    if (selectedMode == ComputeMode::Cuda && !cudaBuilt)
    {
        std::cout << "mode=cuda requested, but cuda backend is not built. Fallback to cpu.\n";
        return ComputeMode::Cpu;
    }
    return selectedMode;
}

#ifdef RIVERFINDER_ENABLE_CUDA
static int cudaDeviceForCurrentThread()
{
    static const int deviceCount = std::max(1, cudaGetVisibleDeviceCount());
    const size_t tidHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return static_cast<int>(tidHash % static_cast<size_t>(deviceCount));
}
#endif



template<int  scale>
std::vector<Res> findBiggestRiver(
    Generator *g,
    int startX, int startZ,
    int sx, int sz,
    int y,
    int min,
    double f = 0.99,
    bool useCuda = false,
    int cudaDeviceId = 0) noexcept
{



    std::vector<Res> result;

    const int W = sx / scale;
    const int H = sz / scale;

    const int stride = W + 1;

#ifdef RIVERFINDER_ENABLE_CUDA
    if (useCuda && scale > 1)
    {
        if (cudaDeviceId < 0)
        {
            cudaDeviceId = cudaDeviceForCurrentThread();
        }
        int maxArea = 0;
        std::vector<CudaRiverCandidate> cudaCandidates;
        if (cudaRingSearchCandidatesFromBiomeNoise(
            &g->bn,
            y,
            W,
            H,
            startX,
            startZ,
            scale,
            min,
            f,
            cudaCandidates,
            maxArea,
            cudaDeviceId))
        {
            result.reserve(cudaCandidates.size());
            for (const auto &candidate: cudaCandidates)
            {
                result.emplace_back(Point{candidate.x, candidate.z}, candidate.area);
            }
            std::sort(result.begin(), result.end(), [](const Res &a, const Res &b) { return a.area > b.area; });
            return result;
        }
        std::cout << "CUDA sampling backend failed, fallback to CPU sampling path.\n";
    }
#endif

    std::vector<int> raw(W * H);
    std::vector<int> prefix((W + 1) * (H + 1), 0);


#define RAW(x,z) raw[(x) + (z)*W]
#define ARR(x,z) prefix[(x) + (z)*stride]

    if constexpr (scale >1)
    {
        for (int z = 0; z < H; z++)
        {
            int worldZ = startZ + z * scale + scale / 2;

            for (int x = 0; x < W; x++)
            {
                int worldX = startX + x * scale + scale / 2;

                RAW(x, z) = value(sampleBiomeNoiseOnRiver(
                    &g->bn, nullptr,
                    worldX / 4,
                    y /4 + 1,
                    worldZ / 4,
                    nullptr,
                    SAMPLE_NO_SHIFT));
            }
        }
    }else
    {
        for (int z = 0; z < H; z++)
        {
            int worldZ = startZ + z;

            for (int x = 0; x < W; x++)
            {
                int worldX = startX + x;

                RAW(x, z) = value(getBiomeAt(g,1,worldX, y, worldZ));
            }
        }
    }

#ifdef RIVERFINDER_ENABLE_CUDA
    if (useCuda)
    {
        if (cudaDeviceId < 0)
        {
            cudaDeviceId = cudaDeviceForCurrentThread();
        }
        int maxArea = 0;
        std::vector<CudaRiverCandidate> cudaCandidates;
        if (cudaRingSearchCandidates(raw.data(), W, H, startX, startZ, scale, min, f, cudaCandidates, maxArea, cudaDeviceId))
        {
            result.reserve(cudaCandidates.size());
            for (const auto &candidate: cudaCandidates)
            {
                result.emplace_back(Point{candidate.x, candidate.z}, candidate.area);
            }
            std::sort(result.begin(), result.end(), [](const Res &a, const Res &b) { return a.area > b.area; });
            return result;
        }
        std::cout << "CUDA backend failed, fallback to CPU path.\n";
    }
#else
    (void) useCuda;
#endif


    for (int z = 1; z <= H; z++)
    {
        for (int x = 1; x <= W; x++)
        {
            ARR(x, z) =
                    RAW(x - 1, z - 1)
                    + ARR(x - 1, z)
                    + ARR(x, z - 1)
                    - ARR(x - 1, z - 1);
        }
    }

    struct RiverArea {
        int area;
        int startX;
        int startZ;

        bool operator<(const RiverArea &other) const noexcept
        {
            if (area != other.area)
                return area < other.area;
            if (startX != other.startX)
                return startX > other.startX;
            return startZ > other.startZ;
        }
    };

    std::priority_queue<RiverArea> pq;

    const int R_out = 128 / scale;
    const int R_in = 24 / scale;

    std::vector<int> dxOut(2 * R_out + 1);
    std::vector<int> dxIn(2 * R_in + 1);

    for (int dz = -R_out; dz <= R_out; dz++)
        dxOut[dz + R_out] =
                (int) floor(sqrt(R_out * R_out - dz * dz));

    for (int dz = -R_in; dz <= R_in; dz++)
        dxIn[dz + R_in] =
                (int) floor(sqrt(R_in * R_in - dz * dz));

    struct Mask {
        int dz;
        int out;
        int in;
    };

    std::vector<Mask> mask;
    mask.reserve(2 * R_out + 1);

    for (int dz = -R_out; dz <= R_out; dz++)
    {
        Mask m;
        m.dz = dz;
        m.out = dxOut[dz + R_out];
        m.in = (abs(dz) <= R_in) ? dxIn[dz + R_in] : -1;
        mask.push_back(m);
    }

    RiverArea max = {0, 0, 0};

    for (int cz = R_out; cz < H - R_out; cz++)
    {
        for (int cx = R_out; cx < W - R_out; cx++)
        {
            int area = 0;

            for (const Mask &m: mask)
            {
                int row = cz + m.dz;

                int L = cx - m.out;
                int R = cx + m.out;

                if (m.in == -1)
                {
                    area +=
                            ARR(R + 1, row + 1)
                            - ARR(L, row + 1)
                            - ARR(R + 1, row)
                            + ARR(L, row);
                } else
                {
                    int Lin = cx - m.in;
                    int Rin = cx + m.in;

                    area +=
                            ARR(Lin, row + 1)
                            - ARR(L, row + 1)
                            - ARR(Lin, row)
                            + ARR(L, row);

                    area +=
                            ARR(R + 1, row + 1)
                            - ARR(Rin + 1, row + 1)
                            - ARR(R + 1, row)
                            + ARR(Rin + 1, row);
                }
            }

            int worldArea = area * scale * scale;

            if (worldArea >= max.area * f && worldArea >= min)
            {
                int worldX = startX + cx * scale;
                int worldZ = startZ + cz * scale;

                pq.push({worldArea, worldX, worldZ});

                if (worldArea > max.area)
                    max.area = worldArea;
            }
        }
    }

    while (!pq.empty())
    {
        if (const auto &ra = pq.top(); ra.area >= max.area * f)
        {
            result.emplace_back( Point{ra.startX,ra.startZ}, ra.area);
            if (f == 1)
                break;
        }
        pq.pop();
    }

    return result;
}

void findBiggestRiverParallelPool(
    ThreadSafeResults<Res> &globalResults,
    Generator *g,
    int startX,
    int startZ,
    int sx,
    int sz,
    int y,
    int minArea,
    bool useCuda = false,
    int numThreads = std::thread::hardware_concurrency()
)
{
    const int chunkSize = 4096 * 2;
    const int overlap = 256;
    std::atomic<int> completedChunks{0};
    int totalChunks = 0;

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastProgressPrintTime = startTime;
    std::mutex progressPrintMutex;
    int x16TopK = 32;
    int x4MaxChecks = 48;
    double x4ContinueFactor = 0.92;

    if (const char *env = std::getenv("RIVERFINDER_X16_TOPK"); env != nullptr)
    {
        const int parsed = std::atoi(env);
        if (parsed > 0)
        {
            x16TopK = std::max(4, std::min(512, parsed));
        }
    }
    if (const char *env = std::getenv("RIVERFINDER_X4_MAX_CHECKS"); env != nullptr)
    {
        const int parsed = std::atoi(env);
        if (parsed > 0)
        {
            x4MaxChecks = std::max(4, std::min(2048, parsed));
        }
    }
    if (const char *env = std::getenv("RIVERFINDER_X4_CONTINUE_FACTOR"); env != nullptr)
    {
        try
        {
            const double parsed = std::stod(env);
            if (parsed >= 0.5 && parsed <= 0.999)
            {
                x4ContinueFactor = parsed;
            }
        } catch (...)
        {
        }
    }

#ifdef RIVERFINDER_ENABLE_CUDA
    if (useCuda)
    {
        std::cout << "CUDA devices visible: " << std::max(1, cudaGetVisibleDeviceCount()) << "\n";
    }
#endif

    struct ChunkTask {
        int x = 0;
        int z = 0;
        int sx = 0;
        int sz = 0;
    };
    std::vector<ChunkTask> tasks;

    auto reportProgress = [&](int completed) {
        std::lock_guard<std::mutex> lock(progressPrintMutex);
        const int percent = int(completed * 100.0 / std::max(1, totalChunks));
        const auto now = std::chrono::high_resolution_clock::now();
        const auto sinceLastMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastProgressPrintTime).count();
        const bool shouldPrint = completed == totalChunks || sinceLastMs >= 1000;
        if (!shouldPrint)
        {
            return;
        }
        lastProgressPrintTime = now;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - startTime).count();

        const double speed = elapsed > 0
            ? static_cast<double>(completed) / static_cast<double>(elapsed) * 1000.0
            : 0.0;
        const double estTotalMs = speed > 0.0
            ? static_cast<double>(totalChunks) / speed * 1000.0
            : 0.0;
        const long long estTotalMsI = static_cast<long long>(estTotalMs);
        const long long remainMs = estTotalMsI - elapsed;

        std::cout << "\rProgress: " << completed << "/" << totalChunks
                << " (" << percent
                << "%)"
                << " - " << speed << " chunks/sec"
                << " - elapsed: " << formatDurationMs(elapsed)
                << " - remaining: " << formatDurationMs(remainMs)
                << " - total(est): " << formatDurationMs(estTotalMsI);
        if (completed == totalChunks)
        {
            std::cout << "\n";
        } else
        {
            std::cout << std::flush;
        }
    };

    auto processChunk = [&](const ChunkTask &task, int assignedCudaDevice) {
        auto blockResultsX16 = findBiggestRiver<16>(
            g,
            startX + task.x,
            startZ + task.z,
            task.sx,
            task.sz,
            y,
            minArea,
            0.8,
            useCuda,
            assignedCudaDevice
        );

        int bx = task.sx / 256 + 2;
        int bz = task.sz / 256 + 2;

        std::vector<Res> flags(bx * bz);
        for (const auto &it: blockResultsX16)
        {
            int x2 = (it.point.x - startX - task.x) / 256;
            int z2 = (it.point.y - startZ - task.z) / 256;
            auto itf = flags[x2 + bx * z2];
            if (itf.area < it.area)
            {
                flags[x2 + bx * z2] = it;
            }
        }

        std::vector<Res> pqX16;
        pqX16.reserve(flags.size());
        for (auto &kv: flags)
        {
            if (kv.area > 0)
            {
                pqX16.push_back(kv);
            }
        }
        auto areaDesc = [](const Res &a, const Res &b) { return a.area > b.area; };
        if (pqX16.size() > static_cast<size_t>(x16TopK))
        {
            std::nth_element(pqX16.begin(), pqX16.begin() + x16TopK, pqX16.end(), areaDesc);
            pqX16.resize(static_cast<size_t>(x16TopK));
        }
        std::sort(pqX16.begin(), pqX16.end(), areaDesc);

        std::unordered_map<uint64_t, Res> blockResultsX4;
        blockResultsX4.reserve(static_cast<size_t>(x16TopK));
        int max = 0;
        int checks = 0;
        int nonImproveStreak = 0;

        for (auto &res: pqX16)
        {
            if (checks >= x4MaxChecks)
            {
                break;
            }
            checks++;
            auto subResults = findBiggestRiver<4>(
                g,
                res.point.x - 256,
                res.point.y - 256,
                512,
                512,
                y,
                minArea,
                1,
                useCuda,
                assignedCudaDevice
            );

            if (subResults.empty())
            {
                nonImproveStreak++;
                if (nonImproveStreak >= 6)
                {
                    break;
                }
                continue;
            }
            if (subResults[0].area < max * x4ContinueFactor)
            {
                nonImproveStreak++;
                if (nonImproveStreak >= 6)
                {
                    break;
                }
                continue;
            }
            nonImproveStreak = 0;

            auto &r = subResults[0];
            uint64_t key = (uint64_t(uint32_t(r.point.x)) << 32) | uint32_t(r.point.y);
            auto it = blockResultsX4.find(key);
            if (it == blockResultsX4.end())
            {
                blockResultsX4.emplace(key, r);
            } else if (it->second.area < r.area)
            {
                it->second = r;
            }
            if (subResults[0].area > max)
            {
                max = subResults[0].area;
            }
        }

        std::vector<Res> filteredResults;
        filteredResults.reserve(blockResultsX4.size());
        for (const auto &entry: blockResultsX4)
        {
            const auto &result = entry.second;
            int relX = result.point.x - (startX + task.x);
            int relZ = result.point.y - (startZ + task.z);
            if (relX > overlap / 2 && relX < task.sx - overlap / 2 &&
                relZ > overlap / 2 && relZ < task.sz - overlap / 2)
            {
                filteredResults.push_back(result);
            }
        }

        std::sort(filteredResults.begin(), filteredResults.end(), [](const Res &a, const Res &b) { return a.area > b.area; });
        if (!filteredResults.empty())
        {
            if (!globalResults.empty() && globalResults.get().area < filteredResults[0].area)
            {
                std::cout << "New Max Found: [" << filteredResults[0].point.x << ", " << filteredResults[0].point.y
                        << "] area: " << filteredResults[0].area << "\n";
            }
            globalResults.addResults(filteredResults);
        }

        int completed = completedChunks.fetch_add(1) + 1;
        reportProgress(completed);
    };

    for (int x = 0; x < sx; x += chunkSize - overlap)
    {
        for (int z = 0; z < sz; z += chunkSize - overlap)
        {
            int currentSx = std::min(chunkSize, sx - x);
            int currentSz = std::min(chunkSize, sz - z);

            if (currentSx >= 256 && currentSz >= 256)
            {
                totalChunks++;
                tasks.push_back({x, z, currentSx, currentSz});
            }
        }
    }

#ifdef RIVERFINDER_ENABLE_CUDA
    if (useCuda)
    {
        const int deviceCount = std::max(1, cudaGetVisibleDeviceCount());
        int workersPerDevice = 1;
        if (const char *env = std::getenv("RIVERFINDER_CUDA_STREAMS_PER_DEVICE"); env != nullptr)
        {
            const int parsed = std::atoi(env);
            if (parsed > 0)
            {
                workersPerDevice = parsed;
            }
        }
        workersPerDevice = std::max(1, std::min(8, workersPerDevice));
        const int workerCount = std::max(1, deviceCount * workersPerDevice);

        std::cout << "CUDA static-shard mode: devices=" << deviceCount
                << ", workers/device=" << workersPerDevice
                << ", total workers=" << workerCount << "\n";

        std::vector<std::vector<std::vector<ChunkTask>>> shards(
            static_cast<size_t>(deviceCount),
            std::vector<std::vector<ChunkTask>>(static_cast<size_t>(workersPerDevice))
        );
        for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex)
        {
            const int deviceId = static_cast<int>(taskIndex % static_cast<size_t>(deviceCount));
            const int workerSlot = static_cast<int>((taskIndex / static_cast<size_t>(deviceCount)) % static_cast<size_t>(workersPerDevice));
            shards[static_cast<size_t>(deviceId)][static_cast<size_t>(workerSlot)].push_back(tasks[taskIndex]);
        }

        std::vector<std::thread> workers;
        workers.reserve(workerCount);

        for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
        {
            for (int workerSlot = 0; workerSlot < workersPerDevice; ++workerSlot)
            {
                const std::vector<ChunkTask> &bucket = shards[static_cast<size_t>(deviceId)][static_cast<size_t>(workerSlot)];
                workers.emplace_back([&, deviceId, &bucket]() {
                    for (const auto &task: bucket)
                    {
                        processChunk(task, deviceId);
                    }
                });
            }
        }
        for (auto &worker: workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    } else
#endif
    {
        ThreadPool pool(numThreads);
        for (const auto &task: tasks)
        {
            pool.enqueue([&, task]() {
                processChunk(task, -1);
            });
        }
    }

    std::cout << "Submitted " << totalChunks << " chunks\n";
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Chunk processing finished in: " << duration.count()
            << "ms for " << totalChunks << " chunks\n";
}
int main(int argc, char **argv)
{
    ComputeMode selectedMode = ComputeMode::Auto;
    int argIndex = 1;
    if (argc > 1)
    {
        const auto parsedMode = parseComputeMode(argv[1]);
        if (parsedMode.has_value())
        {
            selectedMode = *parsedMode;
            argIndex = 2;
        } else
        {
            std::cout << "Invalid mode: " << argv[1] << ". Supported modes: auto/cpu/cuda\n";
            return 1;
        }
    } else
    {
        std::string modeInput = "auto";
        std::cout << "mode(auto/cpu/cuda, default auto): ";
        std::cin >> modeInput;
        if (const auto parsedMode = parseComputeMode(modeInput); parsedMode.has_value())
        {
            selectedMode = *parsedMode;
        } else
        {
            std::cout << "Invalid mode: " << modeInput << ". Supported modes: auto/cpu/cuda\n";
            return 1;
        }
    }

    const ComputeMode activeMode = resolveComputeMode(selectedMode);
    std::cout << "active mode: " << (activeMode == ComputeMode::Cuda ? "cuda" : "cpu") << std::endl;
    const bool useCudaBackend = (activeMode == ComputeMode::Cuda);
    int cudaDeviceCount = 1;
#ifdef RIVERFINDER_ENABLE_CUDA
    if (useCudaBackend)
    {
        cudaDeviceCount = std::max(1, cudaGetVisibleDeviceCount());
    }
#endif

    int d;
    int64_t seed = -8180004378910677489;
    int startX;
    int startZ ;
    int px = 0, pz = 0,y = -62;

    if (argIndex < argc)
    {
        if (!parseNumberArg(argv[argIndex], seed))
        {
            std::cout << "Invalid seed: " << argv[argIndex] << "\n";
            return 1;
        }
        argIndex++;
    } else
    {
        std::cout <<"seed: ";
        std::cin >> seed;
    }

    if (argIndex < argc)
    {
        if (!parseNumberArg(argv[argIndex], px))
        {
            std::cout << "Invalid center_x: " << argv[argIndex] << "\n";
            return 1;
        }
        argIndex++;
    } else
    {
        std::cout <<"center_x: ";
        std::cin >> px;
    }

    if (argIndex < argc)
    {
        if (!parseNumberArg(argv[argIndex], pz))
        {
            std::cout << "Invalid center_z: " << argv[argIndex] << "\n";
            return 1;
        }
        argIndex++;
    } else
    {
        std::cout <<"center_z: ";
        std::cin >> pz;
    }

    if (argIndex < argc)
    {
        if (!parseNumberArg(argv[argIndex], y))
        {
            std::cout << "Invalid y: " << argv[argIndex] << "\n";
            return 1;
        }
        argIndex++;
    } else
    {
        std::cout <<"y: ";
        std::cin >> y;
    }

    if (argIndex < argc)
    {
        if (!parseNumberArg(argv[argIndex], d))
        {
            std::cout << "Invalid r: " << argv[argIndex] << "\n";
            return 1;
        }
        argIndex++;
    } else
    {
        std::cout <<"r: ";
        std::cin >> d;
    }

    if (argIndex < argc)
    {
        std::cout << "Warning: extra arguments ignored:";
        for (int i = argIndex; i < argc; ++i)
        {
            std::cout << " " << argv[i];
        }
        std::cout << "\n";
    }

    startX = px - d;
    startZ = pz - d;
    int xRange = 2*d;
    int zRange = 2*d;


    int minArea = 40000;
    const char *outFile = "out1.txt";
    int outLimit = 1000;


    Generator g;
    setupGenerator(&g, MC_1_21_3, FORCE_OCEAN_VARIANTS);

    applySeed(&g, DIM_OVERWORLD, seed);

    ThreadSafeResults<Res> globalResults;

    findBiggestRiverParallelPool(globalResults, &g, startX, startZ, xRange, zRange, y, minArea, useCudaBackend);

    auto res = globalResults.getAllResults();

    int max = 0;

    std::vector<Res> finallyResults;
    int finalPassIndex = 0;

    for (const auto& it: res)
    {
        const int assignedCudaDevice = useCudaBackend ? (finalPassIndex % cudaDeviceCount) : 0;
        auto temp = findBiggestRiver<1>(
            &g,
            it.point.x - 128 - 32,
            it.point.y - 128 - 32,
            256 + 64,
            256 + 64,
            y,
            1,
            1,
            useCudaBackend,
            assignedCudaDevice
        );
        finalPassIndex++;
        if (!temp.empty() && temp[0].area > max * 0.90)
        {
            finallyResults.push_back(temp[0]);
            if (temp[0].area > max)
            {
                max = temp[0].area;
            }
        }else
        {
            break;
        }
    }

    std::sort(finallyResults.begin(), finallyResults.end(), [](const Res& a, const Res& b) { return a.area > b.area; });

    FILE *fp = fopen(outFile, "w");
    if (fp)
    {
        fprintf(fp, "River Analysis Results (Seed: %lld)\n", (long long) seed);
        fprintf(fp, "Search Area: X=%d to %d, Z=%d to %d\n",
                startX, startX + xRange, startZ, startZ + zRange);
        fprintf(fp, "========================================\n");
    }

    int count = 0;
    for (const auto &it: finallyResults)
    {
        std::cout << "x:" << it.point.x << " y:" << it.point.y
                << "  Area:" << it.area << std::endl;

        if (fp)
        {
            fprintf(fp, "x:%d y:%d  Area:%d\n",
                    it.point.x, it.point.y, it.area);
        }
        count++;
        if (count > outLimit)
        {
            break;
        }
    }
    if (fp)
    {
        fprintf(fp, "\nTotal rivers found: %d\n", count);
        fclose(fp);
        std::cout << "\nResults saved to: " << outFile << std::endl;
    }
    return 0;
}
