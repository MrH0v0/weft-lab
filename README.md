# Weft Lab

一个能在 Blender 3D 视口里直接“上手撕”的针织袜面交互插件。

按住鼠标轻拽，能看到织物从皮肤表面被拉起。继续往外拉，袜面会顺着针织纹路漏针、开孔。松手后孔缘会自然回弹，你还可以抓着旧孔旁边的料子接着撕，或者把两个孔之间剩下的布桥拉断。切到湿态模式后，还能往袜面上加水，观察水膜扩散、织物吸水以及重新贴合皮肤的过程。

![丽莎与松手后的袜面近景，孔边能看见针圈和游离细纱](docs/preview.jpg)

下面的短片截取了拉断布桥和松手的过程。1080p 完整版从第一次轻拉开始，最后绕到孔岸侧面看纱线反光。画面使用真实视口抓取产生的网格，在 Blender EEVEE 中逐帧渲染。[下载完整短片](https://github.com/MrH0v0/weft-lab/releases/download/v0.4.39/weft-lab-lisa-showcase-0.4.39.mp4) · [查看 2560 × 1440 孔岸原图](docs/yarn-detail.jpg)

![两处缺口之间的布桥被拉断，松手后仍然贯通](docs/lisa-knit.webp)

演示角色为《原神》丽莎。模型版权归 miHoYo，模型编辑为神帝宇。

## 安装

当前版本 **0.4.39**，已在 **Windows x64 + Blender 5.3.0** 下验证通过。插件要求 Blender 5.3.0 及以上，其他系统暂未实测。

1. 在 **Releases → Assets** 下载插件包 `weft_replay-0.4.39.zip`。想跟着丽莎演示试撕，再下载 `weft-lab-lisa-demo-0.4.39.zip`。[前往下载](https://github.com/MrH0v0/weft-lab/releases/tag/v0.4.39)
2. 打开 Blender，进入 **编辑 → 偏好设置 → 获取扩展**，点右上角菜单选择 **从磁盘安装**，选中插件 ZIP 并启用。
3. 解压丽莎演示工具和官方丽莎 1.4 模型包。在工具目录运行下面的命令，把第一个路径换成自己的 `丽莎.pmx`。这会在本机生成带完整角色、已绑定右袜和屈膝动作的工程。[官方模型包](https://activity.hdslb.com/blackboard/static/20210316/4d40bc4f98f94fbc71c235832ce3efd4/QIoO5EgUPA.zip) · [演示生成说明](examples/lisa/README.md)

   ```console
   blender --background --python-exit-code 1 --python make_lisa.py -- "模型/丽莎/丽莎.pmx" "weft-lab-lisa.blend"
   ```

4. 打开生成的 `weft-lab-lisa.blend`，鼠标悬停在 3D 视口内按 `N` 展开侧栏，切到 **Weft Lab** 标签页即可。

## 上手试撕

**干态体验**

1. 选择 **干态抓取**，把鼠标移到丽莎右腿有织纹的区域，按住左键先轻轻往外拖，再继续拉远。
2. 看到开孔后松手，下一把可以把抓点放在孔尖旁还有织纹的位置接着撕。
3. 撕出两个相邻的孔后，去拽中间剩下的布料，可以看到布桥一点点变窄，直到彻底拉断贯通。

**湿态与材质调节**

1. 点 **结束实验并恢复原袜** 重置，然后切换到 **湿态抓取 / 泼水**。
2. 鼠标悬停在袜面上按 `W` 加水，再抓住湿润的部位拖动、松手。
3. 面板里可以切换黑色、烟灰、肤色、白色四种配色，并分别调节 **光学厚度** 和 **现场织纹尺寸**。

## 用在自己的角色上

如果想把这套效果套在自己的模型上，先确认**袜面和皮肤网格由同一套骨架驱动**，并且**按米建模**。

1. 在 **半自动袜面绑定** 中分别指定袜面和皮肤网格。
2. 点 **在袜面点选抓点**，选中脚踝附近的袜面后，点击 **生成回放副本**。
3. 点 **干态抓取** 可以固定姿势，仔细对比不同撕法下的孔形。点 **随骨架动作抓取** 可以一边播放角色动画，一边实时抓放。

袜长调整、动作贴肤准备以及探针的详细用法见[使用说明](extension/weft_replay/USER_GUIDE.md)。

## 保存与重开

| 保存时的状态 | 重开工程后可以做什么 |
| --- | --- |
| 有缺口的干袜 | 保留当前的孔缘形态，在原姿势下点 **接着撕当前缺口** 即可继续抓撕 |
| 湿袜 | 保留当时的孔形、湿膜和液量读数，恢复原袜后可开始下一次实验 |
| 固定缓存回放 | 直接拖动时间轴，逐帧查看已经烘焙好的动作 |

> 保存工程会停掉当前的实时求解，把这一刻的袜面状态“定格”下来。实时鼠标轨迹和水量演化不会自动烘焙成动画。

## 从源码构建

在仓库根目录用 **Python 3.11+** 运行以下命令即可打包并校验。

```console
python tools/build_extension.py
blender --command extension validate dist/weft_replay-0.4.39.zip
```

安装包会写入 `dist/`。Windows x64 原生几何模块的 C++ 源码和编译命令见[原生模块构建说明](tools/BUILD.md)。

## 许可与致谢

Weft Lab 贡献者编写的源码采用 **MIT**，包括针织与撕裂算法、原生几何内核和构建脚本。包含 Blender Python API 绑定的完整扩展同时按 **GPL-3.0-or-later** 提供。[MIT 许可](LICENSE) · [扩展 GPL 许可](extension/weft_replay/LICENSE.txt) · [Blender 许可说明](https://www.blender.org/support/faq/)

丽莎模型与角色画面遵循原模型的使用规则。公开演示工具从使用者本机的官方模型生成工程。模型附带的《使用规则.txt》规定“请勿二配”。工具附带的 PMX 读取器来自 MMD Tools。插件内置脚踝回放的人体几何源自 MakeHuman hm08，撕裂声取自 Joseph Sardin 的录音，二者的原始素材均为 CC0。[丽莎模型活动页](https://www.bilibili.com/blackboard/activity-CCayRh076B.html) · [第三方声明](extension/weft_replay/THIRD_PARTY_NOTICES.md)
