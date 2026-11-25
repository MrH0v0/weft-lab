# 构建 Windows x64 原生模块

安装包及源码树内已附 `contact_index_win64.dll`。如需修改几何或接触内核，可使用 Zig 0.14.1，从仓库根目录执行下列命令。五个 `.inc` 实现由 `contact_index.cpp` 直接包含。

```console
zig c++ -O3 -std=c++17 -ffp-contract=off -target x86_64-windows-gnu -shared -static -s extension/weft_replay/contact_index.cpp -o extension/weft_replay/contact_index_win64.dll
python tools/build_extension.py
blender --command extension validate dist/weft_replay-0.4.39.zip
```

Python 打包脚本使用标准库。仓库内已有完整的针织序列和量化回放数据，打包时直接使用。Blender 安装提供运行所需的 Python、NumPy、着色和音频模块。[Zig 0.14.1 源码](https://github.com/ziglang/zig/tree/0.14.1)
