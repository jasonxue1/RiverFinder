#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#include "cubiomes/generator.h"
#include "cubiomes/generator.h"
#include "biomes.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include  <queue>
#include <iostream>
#include <chrono>
#include "Thread.h"
#include  <cmath>

#include "cubiomes/finders.h"

#define AREA_SIZE 256// 搜索步长（不重叠）

Generator g;

struct ForestSize {
    int x = 0;
    int z = 0;
    int dx = 0;
    int dy = 0;
    int dz = 0;
    bool operator<(const ForestSize& other) const {
        // 首先按面积降序，面积相同时按坐标排序
        return dx*dy*dz < other.dx*other.dz*other.dy;
    }
};


std::vector<ForestSize> findForest(Generator * g,long long beginX,long long beginZ,long long endX,long long endZ,uint64_t seed,long long MaxCount = 100) {

    Piece piece[1000];
    std::vector<ForestSize> results;
    std::priority_queue<ForestSize> pq;
    long long beginRegionX =(int) (std::floor(beginX/432)+0.0000001);
    long long beginRegionZ =(int) (std::floor(beginZ/432)+0.0000001);
    long long endRegionX =(int) (std::ceil(endX/432)+0.0000001);
    long long endRegionZ =(int) (std::ceil(endZ/432)+0.0000001);
    for (long long regionX = beginRegionX; regionX <= endRegionX; regionX++) {
        for (long long regionZ = beginRegionZ; regionZ <= endRegionZ; regionZ++) {
            Pos pos;
            // 检查该区块是否有要塞
            if (getStructurePos(Fortress, MC_1_21, seed, regionX, regionZ, &pos)) {
                if ( beginX<=pos.x && beginZ<=pos.z &&endX>=pos.x && endZ>=pos.z &&

                isViableStructurePos(Fortress, g, pos.x, pos.z, 0) ){

                    int n = getFortressPieces(piece,1000,MC_1_21,seed,pos.x/16,pos.z/16);

                    int minX = INT32_MAX;
                    int minZ = INT32_MAX;
                    int maxX = INT32_MIN;
                    int maxZ = INT32_MIN;
                    int minY = INT32_MAX;
                    int maxY = INT32_MIN;
                    for (int i = 0; i < n; i++) {
                        if (piece[i].bb0.x<minX) minX = piece[i].bb0.x;
                        if (piece[i].bb0.x>maxX) maxX = piece[i].bb0.x;
                        if (piece[i].bb1.x<minX) minX = piece[i].bb1.x;
                        if (piece[i].bb1.x>maxX) maxX = piece[i].bb1.x;
                        if (piece[i].bb0.z<minZ) minZ = piece[i].bb0.z;
                        if (piece[i].bb0.z>maxZ) maxZ = piece[i].bb0.z;
                        if (piece[i].bb1.z<minZ) minZ = piece[i].bb1.z;
                        if (piece[i].bb1.z>maxZ) maxZ = piece[i].bb1.z;
                        if (piece[i].bb1.y<minY) minY = piece[i].bb1.y;
                        if (piece[i].bb1.y>maxY) maxY = piece[i].bb1.y;

                    }
                    int dx = maxX - minX;
                    int dz = maxZ - minZ;
                    int dy = maxY - minY;
                    if (dx>=240 && dz>=240&&dy>=39) {
                        pq.emplace((ForestSize{pos.x,pos.z,dx,dy,dz}));
                    }

                }
            }
        }
    }

    int count = 0;
    while (!pq.empty()) {
        auto& ra = pq.top();

        results.emplace_back(ra);
        count++;
        if (count>MaxCount) {
            break;
        }
        pq.pop();


    }
    return results;

}

void findBiggestRiverParallelPool(
    ThreadSafeResults<ForestSize> & globalResults,
    Generator* g,
    int startX,
    int startZ,
    int sx,
    int sz,
    int resCount,
    int numThreads = std::thread::hardware_concurrency()
) {



    ThreadPool pool(numThreads);

    const int chunkSize = 432*100;

    std::atomic<int> completedChunks{0};
    int totalChunks = 0;

    auto startTime = std::chrono::high_resolution_clock::now();

    // 提交所有任务到线程池
    for (int x = 0; x < sx; x += chunkSize ) {
        for (int z = 0; z < sz; z += chunkSize ) {
            int currentSx = std::min(chunkSize, sx - x);
            int currentSz = std::min(chunkSize, sz - z);

            if (currentSx >= 432 && currentSz >= 432) {
                totalChunks++;

                // 为每个块创建独立的任务
                pool.enqueue([&, x, z, currentSx, currentSz]() {
                    // 创建线程本地Generator
                    Generator localG;
                    memcpy(&localG, g, sizeof(Generator));

                    auto blockResults = findForest(
                        &localG,
                          startX + x,
                        startZ + z ,
                        startX +x + currentSx,
                        startZ +z+ currentSz,
                        g->seed,
                        1000
                    );

                    globalResults.addResults(blockResults);

                    // 更新进度
                    int completed = completedChunks.fetch_add(1) + 1;
                    if (completed % 10 == 0) {
                        auto currentTime = std::chrono::high_resolution_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                            currentTime - startTime
                        ).count();
                        double speed = static_cast<double>(completed) / elapsed * 1000;
                        std::cout << "Progress: " << completed << "/" << totalChunks
                                  << " chunks (" << static_cast<int>(completed * 100.0 / totalChunks)
                                  << "%) - " << speed << " chunks/sec\n";
                    }
                });
            }
        }
    }

    std::cout << "Submitted " << totalChunks << " chunks to thread pool\n";

    // 等待所有任务完成（线程池析构时会自动等待）
    // ThreadPool析构时会自动等待所有任务完成

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime
    );

    std::cout << "Thread pool processing took: " << duration.count()
              << "ms for " << totalChunks << " chunks\n";
    return;
}








template<typename Func>
void measure_time(Func func, const std::string& name = "Function") {
    auto start = std::chrono::high_resolution_clock::now();

    // 执行传入的函数
    func();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << name << " took: "
              << duration.count() << " microseconds ("
              << duration.count() / 1000.0 << " ms)" << std::endl;
}



/* ================== 主程序 ================== */









int main1(int argc, char **argv)
{


    uint64_t seed = 7103583996044705691;



    setupGenerator(&g, MC_1_21, FORCE_OCEAN_VARIANTS);

    applySeed(&g, DIM_NETHER, seed);

    int startX, startZ;
    int endX, endZ;

   // std::cout<<"请输入种子：\n";
    //std::cin>>seed;

    std::cout<<"请输入查找初始坐标: \n";
    std::cin>>startX>>startZ;

    std::cout<<"请输入查找末坐标: \n";
    std::cin>>endX>>endZ;
    if (startX>endX) {
        std::swap(startX, endX);
    }
    if (startZ>endZ) {
        std::swap(startZ, endZ);
    }


    std::vector<ForestSize> forestSizes;

    int regionX, regionZ;

    int maxArea = 0;

    ThreadSafeResults<ForestSize> globalResults;

    // 使用从命令行获取的参数
    findBiggestRiverParallelPool(globalResults, &g,startX,startZ,endX-startX,endZ-startZ ,100);


    // 输出到文件（如果需要）
    FILE* fp = fopen("outData.txt", "w");
    if (fp) {
        fprintf(fp, "Nether Fortress Finder.. (Seed: %lld)\n", (long long)seed);
        fprintf(fp, "Search Area: X=%d to %d, Z=%d to %d\n",
                startX, endX, startZ, endZ);
        fprintf(fp, "========================================\n");
    }

    int count = 0;
    for (const auto& it : globalResults.getAllResults()) {
        std::cout << "x:" << it.x << " z:" << it.z
                  << "  size: (" <<it.dx << " ," <<it.dy<< " ," <<it.dz <<") = ("<< it.dx*it.dy*it.dz<<")"<< std::endl;

        if (fp) {
            fprintf(fp, "x:%d y:%d size: (%d, %d, %d) | (%d)\n",
                    it.x,it.z,it.dx,it.dy,it.dz,it.dx*it.dy*it.dz);
        }
        count++;
        if (count >1000) {
            return 0;
        }
    }

    if (fp) {
        fprintf(fp, "\nTotal rivers found: %d\n", count);
        fclose(fp);
        std::cout << "\nResults saved to: " << "outData.txt" << std::endl;
    }



    return 0;
}
