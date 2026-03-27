cmake_minimum_required(VERSION 3.9)
project(riverFinder)
option(ENABLE_CUDA_MODE "Enable CUDA mode selection (runtime mode can select cuda)" OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
    set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    # 类似处理 C++ 标志
else()
	set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
	set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

if(WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_WIN32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32")
endif()
set(CMAKE_CXX_STANDARD 20)

set(HEADERS
		cubiomes/finders.h
		cubiomes/generator.h
		cubiomes/layers.h
		cubiomes/biomenoise.h
		cubiomes/biomes.h
		cubiomes/noise.h
		cubiomes/rng.h
		cubiomes/util.h
		cubiomes/quadbase.h
		Thread.h
)
set(SOURCES
		cubiomes/finders.c
		cubiomes/generator.c
		cubiomes/layers.c
		cubiomes/biomenoise.c
		cubiomes/biomes.c
		cubiomes/noise.c
		cubiomes/util.c
		cubiomes/quadbase.c
		river_finder.cpp
)

if(ENABLE_CUDA_MODE)
	enable_language(CUDA)
	list(APPEND SOURCES cuda_accel.cu)
endif()

add_executable(riverFinder ${SOURCES})

if(ENABLE_CUDA_MODE)
    target_compile_definitions(riverFinder PRIVATE RIVERFINDER_ENABLE_CUDA)
	set_target_properties(riverFinder PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(riverFinder PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --prec-div=true --prec-sqrt=true>
    )
endif()
