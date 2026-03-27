# 河流搜索程序

基于的cubiomes库

只能搜1.18+

## 使用方法

编译后，依次输入模式、种子、初始 x、初始 y、初始 z、搜索半径 r（搜索区域为 `2*r` 的方形）。

支持模式：
- `auto`：自动选择，若编译时开启 CUDA 模式则优先使用 CUDA，否则使用 CPU
- `cpu`：强制 CPU
- `cuda`：强制 CUDA，若当前程序未编译 CUDA 支持会自动回退到 CPU

注意：当前 CUDA 路径会加速环形面积统计计算；并且在 `scale>1` 路径下，河流采样会在 GPU 执行。
CUDA 模式下会自动查询可见 GPU 数量，并在并行任务中按线程分配设备（可同时利用多张卡）。

也可以在命令行第一个参数直接指定模式，例如：

```bash
./riverFinder auto
./riverFinder cpu
./riverFinder cuda
```

也支持直接传入参数（无需交互）：

```bash
./riverFinder cuda <seed> <center_x> <center_z> <y> <r>
```

如果你有多卡（例如 4 张 H200），可通过 `CUDA_VISIBLE_DEVICES` 控制参与计算的设备：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./riverFinder cuda
```

如需在 CMake 编译时启用 CUDA 模式选择宏：

```bash
cmake -S . -B build -DENABLE_CUDA_MODE=ON
cmake --build build
```
