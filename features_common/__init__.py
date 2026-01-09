"""统一的特征读取与数据集工具。

这里的目标很简单：让四个模型导出的 .pt（dict / dataclass / tensor）都能用同一套 API 读起来。

约定（你当前的导出范式）
- 推荐保存 dict：
  - per_frame_features: Tensor[W,T,Hf,Wf,C]
  - frame_paths: List[List[str]]
  - meta: dict
  - （可选）features: Tensor[W,Hf,Wf,C] 或其它

本包不会依赖任何模型代码，不会触发联网。
"""
