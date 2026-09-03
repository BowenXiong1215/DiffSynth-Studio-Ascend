# DiffSynth-Studio 昇腾 910B 补丁包

此补丁包将官方 DiffSynth-Studio 源码转换为已经验证过的昇腾 910B Wan2.2 LoRA 版本。

## 基线

```text
仓库：https://github.com/modelscope/DiffSynth-Studio
Commit：4dbf980d4d0eb34eda136300dd0d72014cff8965
版本：2.1.6
```

安装器处理 15 个官方源码文件，并加入 16 个昇腾环境、测试、训练、下载、验收和说明文件。官方源码文件通过 `sed -i` 更新；新增文件从补丁载荷复制。

## 使用

下载官方源码：

```bash
wget -O DiffSynth-Studio.tar.gz \
  https://github.com/modelscope/DiffSynth-Studio/archive/4dbf980d4d0eb34eda136300dd0d72014cff8965.tar.gz

tar -xzf DiffSynth-Studio.tar.gz
mv DiffSynth-Studio-4dbf980d4d0eb34eda136300dd0d72014cff8965 DiffSynth-Studio
```

解压本补丁包，并执行：

```bash
bash diffsynth-studio-ascend-910b-patch-20260903/install.sh \
  /absolute/path/to/DiffSynth-Studio
```

成功标志：

```text
DiffSynth-Studio Ascend 910B patch: PASS
```

独立校验：

```bash
bash diffsynth-studio-ascend-910b-patch-20260903/verify.sh \
  /absolute/path/to/DiffSynth-Studio
```

安装脚本可以重复执行；已经应用的文件会显示 `present`。

## 安装后测试

```bash
cd /absolute/path/to/DiffSynth-Studio
python tests/test_ascend_wan_rope.py
python scripts/ascend_npu_smoke.py
```

完整镜像、模型下载、训练和验收流程见：

```text
docs/zh/Wan2.2_Ascend_910B_LoRA_Quickstart.md
```
