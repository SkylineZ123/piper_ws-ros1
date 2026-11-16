#!/bin/bash
# 只配置指定 USB 路径上的 Piper CAN 接口，其他 CAN 接口保持原样。
# 用法：sudo bash can_config.sh [接口名] [波特率] [USB 路径]
# 默认：piper_left 1000000 1-6.1:1.0
# 预览：bash can_config.sh --dry-run（不需要 sudo，不修改网络接口）
set -Eeuo pipefail

usage() {
    cat <<'HELP'
用法：bash can_config.sh [接口名] [波特率] [USB 路径] [--dry-run]
默认：piper_left 1000000 1-6.1:1.0
仅配置匹配 USB 路径的 CAN 接口，不要求系统只有一个 CAN 接口。
--dry-run  只显示匹配结果和将执行的命令，不修改接口。
HELP
}

DRY_RUN=false
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "错误：未知选项 $arg" >&2; exit 1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
if ((${#POSITIONAL[@]} > 3)); then usage >&2; exit 1; fi
TARGET_NAME="${POSITIONAL[0]:-piper_left}"
TARGET_BITRATE="${POSITIONAL[1]:-1000000}"
USB_ADDRESS="${POSITIONAL[2]:-1-6.1:1.0}"

if [[ ! "$TARGET_NAME" =~ ^[[:alnum:]_.-]{1,15}$ || "$TARGET_NAME" == . || "$TARGET_NAME" == .. ]]; then
    echo "错误：接口名必须是 1–15 个字母、数字、下划线、点或连字符。" >&2; exit 1
fi
if [[ ! "$TARGET_BITRATE" =~ ^[1-9][0-9]{0,6}$ ]] || ((TARGET_BITRATE > 1000000)); then
    echo "错误：Piper CAN 波特率应为 1–1000000 的整数。" >&2; exit 1
fi
for tool in ip readlink awk; do
    if ! command -v "$tool" >/dev/null; then echo "错误：缺少命令 $tool" >&2; exit 1; fi
done

MATCHES=()
for device in /sys/class/net/*; do
    [[ -r "$device/type" ]] || continue
    [[ "$(<"$device/type")" == 280 ]] || continue
    iface="${device##*/}"
    device_path=$(readlink -f "$device/device") || device_path=""
    bus_info="${device_path##*/}"
    if [[ "$bus_info" == "$USB_ADDRESS" ]]; then
        MATCHES+=("$iface")
    else
        printf '忽略 CAN 接口：%s（USB：%s）\n' "$iface" "${bus_info:-未知}" >&2
    fi
done
if ((${#MATCHES[@]} != 1)); then
    printf '错误：USB 路径 %s 匹配到 %d 个 CAN 接口，必须恰好匹配一个；未修改任何接口。\n' "$USB_ADDRESS" "${#MATCHES[@]}" >&2
    exit 1
fi
INTERFACE_NAME="${MATCHES[0]}"
if [[ "$INTERFACE_NAME" != "$TARGET_NAME" && -e "/sys/class/net/$TARGET_NAME" ]]; then
    echo "错误：名称 $TARGET_NAME 已被其他接口占用；未修改任何接口。" >&2
    exit 1
fi
printf '目标：%s（USB：%s）→ %s，波特率 %s bit/s\n' "$INTERFACE_NAME" "$USB_ADDRESS" "$TARGET_NAME" "$TARGET_BITRATE"

if $DRY_RUN; then
    printf '预览：ip link set dev %q down\n' "$INTERFACE_NAME"
    printf '预览：ip link set dev %q type can bitrate %q\n' "$INTERFACE_NAME" "$TARGET_BITRATE"
    if [[ "$INTERFACE_NAME" != "$TARGET_NAME" ]]; then
        printf '预览：ip link set dev %q name %q\n' "$INTERFACE_NAME" "$TARGET_NAME"
    fi
    printf '预览：ip link set dev %q up\n' "$TARGET_NAME"
    exit 0
fi

# 只读检查；接口已经按目标配置时不打断现有连接。
DETAILS=$(ip -details -o link show dev "$INTERFACE_NAME")
CURRENT_BITRATE=$(awk '{for (i=1;i<NF;i++) if ($i=="bitrate") {print $(i+1); exit}}' <<< "$DETAILS")
FLAGS=$(<"/sys/class/net/$INTERFACE_NAME/flags")
if [[ "$INTERFACE_NAME" == "$TARGET_NAME" && "$CURRENT_BITRATE" == "$TARGET_BITRATE" ]] && ((FLAGS & 1)); then
    echo "接口 $TARGET_NAME 已启动，波特率为 $TARGET_BITRATE，无需修改。"
    exit 0
fi

PRIVILEGE=()
if ((EUID != 0)); then
    if ! command -v sudo >/dev/null; then echo '错误：配置网络接口需要 root，请使用 sudo 运行。' >&2; exit 1; fi
    sudo -v
    PRIVILEGE=(sudo)
fi
trap 'echo "错误：CAN 配置未完成，请检查上方命令错误和接口状态。" >&2' ERR
"${PRIVILEGE[@]}" ip link set dev "$INTERFACE_NAME" down
"${PRIVILEGE[@]}" ip link set dev "$INTERFACE_NAME" type can bitrate "$TARGET_BITRATE"
if [[ "$INTERFACE_NAME" != "$TARGET_NAME" ]]; then
    "${PRIVILEGE[@]}" ip link set dev "$INTERFACE_NAME" name "$TARGET_NAME"
fi
"${PRIVILEGE[@]}" ip link set dev "$TARGET_NAME" up
printf '已配置：%s，USB %s，波特率 %s bit/s。其他 CAN 接口未修改。\n' "$TARGET_NAME" "$USB_ADDRESS" "$TARGET_BITRATE"
