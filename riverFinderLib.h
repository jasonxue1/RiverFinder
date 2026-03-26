//
// Created by zhdds on 2026/3/15.
//

#ifndef RIVERFINDER_RIVERFINDERLIB_H
#define RIVERFINDER_RIVERFINDERLIB_H

#endif //RIVERFINDER_RIVERFINDERLIB_H

#ifdef RIVERFINDER_EXPORTS
#define RIVERFINDER_API __declspec(dllexport)
#else
#define RIVERFINDER_API __declspec(dllimport)
#endif

#ifdef __cplusplus
 extern "C" {
#endif

/**
 * Single-phase river search.
 * @param seed world seed
 * @param startX region start X
 * @param startZ region start Z
 * @param width region width
 * @param height region height
 * @param y search height (use -60)
 * @param minArea minimum river area
 * @param numThreads thread count (0 = use setThreadCount or default)
 * @return [x1, z1, area1, x2, z2, area2, ...] or null.
 */
RIVERFINDER_API int *riverSearch(long long seed, int startX, int startZ,
                 int width, int height, int y, int minArea, int numThreads);

/**
 * Reserved: single-phase progress query. Returns [current, total] or null if not implemented.
 */
/**/
RIVERFINDER_API int *getSearchProgress();



#ifdef __cplusplus
}


#endif
