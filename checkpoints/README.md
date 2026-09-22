# 模型权重

本目录用于经登记的共享参考权重。既有权重保留在原实验包中，避免丢失配套配置、日志和指标。

- 既有权重精确位置：[artifact_index.json](artifact_index.json)，按原路径和大小登记；不重复复制大文件。
- 模型定义：[model.py](../research_code/dae_pipeline/model.py)。
- 新的独立共享权重可放本目录具名子目录，并记录架构、CR、训练协议、来源运行和代码版本。
- 历史 pickle 模块名兼容由 [runtime_paths.py](../runtime_paths.py) 提供；独立读取旧对象前先调用 `install_legacy_imports()`。优先使用已有读取入口。
- 静态基线缓存中的旧源码指纹通过迁移表登记的精确等价对兼容；原 `plot_data.pkl` 不重写，未登记的代码或协议变化仍拒绝复用。
