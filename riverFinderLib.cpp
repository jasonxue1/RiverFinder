//
// Created by zhdds on 2026/3/15.
//

#include "riverFinderLib.h"

#include <cstring>

#include "river_finder.cpp"



pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

inline int * riverSearch(long long seed, int startX, int startZ, int width, int height, int y, int minArea, int numThreads)
{
    std::memset((void*) &progress,0,sizeof(progress));
    Generator g;
    setupGenerator(&g, MC_1_21_3, FORCE_OCEAN_VARIANTS);

    applySeed(&g, DIM_OVERWORLD, seed);

    ThreadSafeResults<Res> globalResults;
    progress.type = 0;
    if (numThreads>0)
    findBiggestRiverParallelPool(globalResults, &g, startX, startZ, width, height,y, minArea,&progress,numThreads);
    else findBiggestRiverParallelPool(globalResults, &g, startX, startZ, width, height,y, minArea,&progress);
    auto res = globalResults.getAllResults();
    int max = 0;
    std::vector<Res> finallyResults;
    for (const auto& it: res)
    {
        auto temp = findBiggestRiver<1>(&g,it.point.x - 128 -32,it.point.y - 128 - 32,256+64,256+64, y, 1,1);
        if (!temp.empty() && temp[0].area > max * 0.8)
        {
            finallyResults.push_back(temp[0]);
            if (temp[0].area > max )
            {
                max = temp[0].area;
            }
        }else
        {
            break;
        }
    }
    std::ranges::sort(finallyResults, [](const Res& a, const Res& b) { return a.area > b.area; });
    int * result =(int* ) malloc(finallyResults.size() * 3 * sizeof(int) + 3);
    int i = 0;
    for (auto& it : finallyResults)
    {
        result[i++] = it.point.x;
        result[i++] = it.point.y;
        result[i++] = it.area;
    }
    result[i++] = 0;
    result[i++] = 0;
    result[i++] = 0;
    return result;
}

inline int * getSearchProgress()
{
    if (progress.total == 0)
    {
        return nullptr;
    }else
    {
        int* res = (int*)malloc(sizeof(progress));
        res[0] = progress.current;
        res[1] = progress.total;
        res[2] = progress.x;
        res[3] = progress.y;
        res[4] = progress.area;
        res[5] = progress.type;
        return res;
    }
}

