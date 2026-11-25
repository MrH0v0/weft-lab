# 干湿针织试穿腿模

打开 `weft-lab-leg-studio-0.4.40.blend` 即可在同一条腿上试干态和湿态。场景自带袜面、拟肤材质、摄影棚灯光和一段屈膝动作。先安装并启用 Weft Lab 插件，再用 Blender 打开工程。[下载工程与插件](https://github.com/MrH0v0/weft-lab/releases/tag/v0.4.40)

在 3D 视口按 `N`，打开 **Weft Lab**。选择 **干态抓取**，在膝盖附近的袜面按住左键，逐渐拉远。松手后可以在孔尖旁续撕，或抓住两个孔之间的布桥。

点 **结束实验并恢复原袜**，切换到 **湿态抓取 / 泼水**。把鼠标移到小腿上，按 `W` 泼水，再按住左键轻拉湿润处，松手看布料贴回腿上。继续拉远可以把湿袜撕开。

## 从源码重新生成

在已启用插件的 Blender 5.3.0 配置下，从仓库根目录运行下面的命令，输出路径使用新的文件名。

```console
blender --background --python-exit-code 1 --python examples/leg/make_leg.py -- weft-leg.blend
```

`make_leg.py` 依据脚趾至大腿的截面生成网格，绑定骨架并布光。`weft-yarn.blend` 提供已烘焙的纱线材质，`yarn_finish.py` 保留法线、纤维方向和粗糙度的烘焙代码。这些示例资源采用仓库的 MIT 许可。
