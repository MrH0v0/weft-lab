# 在本机生成丽莎演示工程

这个工具把《原神》丽莎 1.4 导入 Blender，保留服装、骨架和表情，给右腿装上可抓取的针织袜面，并添加站姿、屈膝动作与摄影棚灯光。模型原始文件由你从官方活动页下载。准备好这些文件后即可离线生成。

## 操作

1. 在 Blender 5.3.0 或更高版本中安装并启用 Weft Lab 0.4.39。
2. 下载官方丽莎 1.4 模型并解压。保留 `丽莎.pmx` 旁的 `Texture` 目录。[原神 1.4 活动页](https://www.bilibili.com/blackboard/activity-CCayRh076B.html) · [活动原版丽莎 ZIP](https://activity.hdslb.com/blackboard/static/20210316/4d40bc4f98f94fbc71c235832ce3efd4/QIoO5EgUPA.zip)
3. 解压发布附件 `weft-lab-lisa-demo-0.4.39.zip`，在包含 `make_lisa.py` 的目录打开终端。将下面的模型路径改为你的实际路径，输出文件选一个新的名字。

   ```console
   blender --background --python-exit-code 1 --python make_lisa.py -- "模型/丽莎/丽莎.pmx" "weft-lab-lisa.blend"
   ```

   Windows PowerShell 找不到 `blender` 命令时，把命令开头的 `blender` 替换为 `& "你的安装目录\blender.exe"`，其余参数相同。

4. 打开生成的工程，在 3D 视口按 **N → Weft Lab → 干态抓取**。先在右膝的织物上轻拉，然后拉远、松手。试湿态时先恢复原袜，点 **湿态抓取 / 泼水**，将指针放在袜面上按 **W**。鼠标滚轮可拉远查看完整角色。

官方 1.4 PMX 的 SHA-256 是 `0b8db0b0c73ab23357f7c5d4e3cc2507eb8a1d32ca871b0760d25ba68ddb6632`。脚本针对这份模型的材质和骨架。重复生成时请更换输出文件名或路径。

## 查看纱线光泽

进入材质预览，靠近右膝并绕腿转动。窄侧灯会扫过针圈，拉伸时可以对照同一处的透肤、纱线方向和孔岸反光。场景里的冷暖两盏条形灯可以分别移动。

`weft-yarn.blend` 是随工具提供的自研材质库。针圈覆盖和高度来自纱线路径与截面。另有两张 1024 × 1024 周期贴图，分别保存外层纤维的切线空间法线，以及纱束方向、粗糙度和覆盖率。材质通过 `Knit` UV 读取这些纹理，跟随拖拽中的袜面形变。[纤维烘焙源码](yarn_finish.py)

## 许可

Weft Lab 编写的生成脚本、纱线材质库和自制纱线贴图采用 MIT。随工具分发的 PMX 读写模块来自 MMD Tools，保留了原作者版权声明。包含该模块的完整演示工具同时按 GPLv3 提供。[MIT 许可](LICENSE.txt) · [MMD Tools 源码版本](https://github.com/MMD-Blender/blender_mmd_tools/blob/e1ceb3e5944acdb3aecc97ff40f54737853ae215/mmd_tools/core/pmx/__init__.py) · [GPLv3 许可](third_party/MMD-Tools-LICENSE.txt)

丽莎角色及模型版权归 miHoYo，模型编辑为神帝宇。预览与短片由 Blender EEVEE 渲染，袜面形变来自项目的视口交互。生成的 `.blend` 包含角色网格与贴图，供本机体验，须遵守官方模型包内《使用规则.txt》关于二次配布和商用的规定。
