# Introduction

按论文三章组织的新实验与合并系统见 [EXPERIMENTS.md](EXPERIMENTS.md)。入口分别为 `experiments.ch03_xpbmpm`、`experiments.ch04_rigid`、`experiments.ch05_morton` 和 `experiments.combined`，共用 `pbmpm/` 数值核心。

使用taichi语言实现的pbmpm流体仿真模拟器

# Install

```txt
taichi: 1.7.4
python: 3.10.11
```



# Version Update

- 2025-12-05：完成对于流体、弹性体、沙子的仿真实现。完成粒子与刚体的碰撞逻辑



# Reference

> *Chris Lewin*. **[A Position Based Material Point Method](https://seed.ea.com/)**. ACM SIGGRAPH 2024.
