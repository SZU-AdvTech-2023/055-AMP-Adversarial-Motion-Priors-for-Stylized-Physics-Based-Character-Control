## 论文复现：AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

### 安装环境

确保运行软件环境为 Linux 操作系统、Python 3.8 编程环境；并确保有插入显示器（或使用具有3D加速的虚拟显示器），以便渲染。

首先下载物理仿真环境 Isaac Gym Preview 4 [点击此处](https://developer.nvidia.com/isaac-gym)，按照其中的说明安装好，然后确保实例程序能够正常运行：`python/examples` ，比如`joint_monkey.py`。

然后，安装如下所需的包：

```
"gym==0.23.1",
"torch",
"omegaconf",
"termcolor",
"jinja2",
"hydra-core>=1.2",
"rl-games>=1.6.0",
"pyvirtualdisplay",
"urdfpy==0.0.22",
"pysdf==0.1.9",
"warp-lang==0.10.1",
"trimesh==3.23.5",
```

### 训练模型

对于训练原始的模型，设置 HumanoidAMPPO.yaml 配置为 `reward_combine: 'add' ` ，然后使用下面的命令行：

```bash
python launch.py task=HumanoidAMP headless=True
```

在训练好以后，测试训练的结果如下：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

对于改进后的模型，设置 HumanoidAMPPO.yaml 配置为 `reward_combine: 'mul' ` ，然后使用下面的命令行：

```bash
python launch.py task=HumanoidAMP headless=True
```

在训练好以后，测试训练的结果如下：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn
```

### 渲染结果到视频

将训练好的模型结果渲染到视频，可以使用如下命令行：

```bash
python launch.py task=HumanoidAMP headless=False test=True num_envs=64 checkpoint=/path/to/saved/model/in/runs/nn capture_video=True
```

所渲染的视频会保存到当前工作目录下。

### 转换为基于动力学的动捕文件

可以参考如下函数（其余详见库 `fmbvh` 中的各个模块）：

`./isaacgymenvs/tasks/amp/utils_amp/motion_lib.export_bvh`

在运行上面的训练、测试等指令，都会在`./runs/` 目录下自动导出动捕文件。

