#!/usr/bin/env bash
set -uo pipefail

REPO="/data16t/yzj/projects/EGIInet"
CKPT="/data16t/yzj/datasets/EGIInet-checkpoints"
OUT="/data16t/yzj/outputs/EGIInet/all_checkpoint_eval"
ENV_SCRIPT="/data16t/yzj/envs/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$OUT"

# 九个官方 checkpoint 对应的类别
CATEGORIES=(cabinet car chair couch lamp plane table watercraft)

# 检查全部权重
for f in "$CKPT/all-eight-ckpt-best.pth"; do
    if [[ ! -f "$f" ]]; then
        echo "缺少权重：$f"
        exit 1
    fi
done

for c in "${CATEGORIES[@]}"; do
    if [[ ! -f "$CKPT/${c}-ckpt-best.pth" ]]; then
        echo "缺少权重：$CKPT/${c}-ckpt-best.pth"
        exit 1
    fi
done

cd "$REPO" || exit 1

# 保存本次运行前的配置，退出时恢复
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$OUT/config_vipc.before_eval_${STAMP}.py"
cp config_vipc.py "$BACKUP"

restore_config() {
    cp "$BACKUP" "$REPO/config_vipc.py"
    echo "已恢复原始配置：$BACKUP"
}
trap restore_config EXIT INT TERM

# 每次测试前，从同一份原始配置生成测试配置
set_config() {
    local weight="$1"
    local category="$2"

    python - "$BACKUP" "$weight" "$category" <<'PY'
from pathlib import Path
import re
import sys

base, weight, category = sys.argv[1:4]
s = Path(base).read_text()

updates = [
    (
        r"^\s*__C\.CONST\.DEVICE\s*=.*$",
        "__C.CONST.DEVICE = '1'",
        "GPU device"
    ),
    (
        r"^\s*#?\s*__C\.CONST\.WEIGHTS\s*=.*$",
        f'__C.CONST.WEIGHTS = r"{weight}"',
        "checkpoint path"
    ),
    (
        r"^\s*__C\.TEST\.CATE\s*=.*$",
        f"__C.TEST.CATE = '{category}'",
        "test category"
    ),
]

for pattern, replacement, label in updates:
    s, n = re.subn(pattern, replacement, s, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(
            f"无法唯一设置 {label}，匹配次数={n}；已停止测试。"
        )

Path("config_vipc.py").write_text(s)
print(f"配置完成：GPU=1, category={category}")
print(f"权重：{weight}")
PY
}

FAILED="$OUT/failed_tests.txt"
: > "$FAILED"

run_test() {
    local model_name="$1"
    local category="$2"
    local weight="$3"
    local log="$OUT/${model_name}_${category}.log"

    echo
    echo "=================================================="
    echo "模型：$model_name"
    echo "测试类别：$category"
    echo "权重：$weight"
    echo "日志：$log"
    echo "=================================================="

    if ! set_config "$weight" "$category"; then
        echo "$model_name,$category,CONFIG_FAILED" >> "$FAILED"
        return
    fi

    # main.py 会读取 config_vipc.py 中的 DEVICE='1'
    if (
        source "$ENV_SCRIPT"
        conda activate /data16t/yzj/envs/egiinet
        cd "$REPO"
        python -u main.py --test
    ) > "$log" 2>&1; then
        if grep -q "TEST RESULTS" "$log"; then
            echo "测试程序已结束，检查指标输出。"
        else
            echo "$model_name,$category,NO_RESULTS" >> "$FAILED"
            echo "警告：进程退出，但日志中没有 TEST RESULTS。"
        fi
    else
        echo "$model_name,$category,TEST_FAILED" >> "$FAILED"
        echo "测试失败，详见：$log"
    fi

    grep -A 4 "TEST RESULTS" "$log" | tail -n 5 || true
}

# 第一组：全八类别联合训练的模型，逐类别评估
for c in "${CATEGORIES[@]}"; do
    run_test "all-eight" "$c" "$CKPT/all-eight-ckpt-best.pth"
done

# 第二组：八个类别专用模型，各自在对应类别上评估
for c in "${CATEGORIES[@]}"; do
    run_test "$c" "$c" "$CKPT/${c}-ckpt-best.pth"
done

# 汇总日志中的 Overall 指标
python - "$OUT" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []

for log in sorted(root.glob("*.log")):
    lines = log.read_text(errors="replace").splitlines()
    result_lines = [
        line for line in lines
        if "Overall" in line and "TEST RESULTS" not in line
    ]

    if not result_lines:
        continue

    line = result_lines[-1]
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", line)

    # Overall 标签本身没有数字，正常应解析出三个指标
    if len(values) >= 3:
        cd_l1, cd_l2, f1 = values[-3:]
        rows.append([
            log.stem, cd_l1, cd_l2, f1, str(log)
        ])

out = root / "summary.csv"
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment", "CDl1", "CDl2", "F1", "log"])
    writer.writerows(rows)

print(f"\n汇总文件：{out}")
print(f"提取到 {len(rows)} 组指标。")
for row in rows:
    print(",".join(row[:4]))
PY

echo
echo "所有批量测试已结束。"
echo "指标汇总：$OUT/summary.csv"
echo "失败记录：$FAILED"
