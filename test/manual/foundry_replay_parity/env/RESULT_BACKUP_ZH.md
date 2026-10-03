# 限时 GPU assignment：先启动本地备份，再启动实验

本轮 `deepep_bank_load_unpatched_full_r2` 运行会话 exit0，38个 rank-phase completion 已在本地日志中保存，但未在 assignment 结束前拉回原始报告。用户确认远端会清空文件，故该报告按丢失处理，不纳入正式性能结论。不能拿 completion 日志代替 raw timing、数值报告或 source manifest。

今后每次限时 assignment 先启动 [pull_results_until_deadline.py](pull_results_until_deadline.py)。脚本仅在本地运行，用已有 Python/SSH/rsync；不安装软件、不写系统配置、不开 cron、不修改远端源代码。

```bash
# 在本地实验目录运行；路径/主机/剩余分钟使用本次真实值。
# known_hosts 必须已经独立核对；不能关闭 host-key 检查。
python3 env/pull_results_until_deadline.py \
  --host user@gpu-host \
  --remote-root /absolute/remote/experiment \
  --output /absolute/local/experiment/live_backup_01 \
  --known-hosts /absolute/local/verified_known_hosts \
  --remaining-minutes 75 \
  > host/live_backup_01.log 2>&1 &
backup_pid=$!
printf '%s\n' "$backup_pid" > host/live_backup_01.pid
```

也可用 `--deadline 2026-10-04T07:27:00Z` 固定截止时间，替代 `--remaining-minutes`。这是示例时间，必须换成实际 assignment 截止时间。不要在实验运行之后才开始倒计时。续时应先正常中止旧的本地 supervisor，再用 `--resume` 和新的正确截止时间重启；它要求源路径、SSH配置和要备份的目录完全相同。

1. **第一轮同步成功后才启动长实验。** 查看 `status.json` 的 `successes` 和 `last.exit_code`。如果是认证失败，先解决备份问题。
2. 默认每30秒同步一次，最后10分钟改为每5秒一次；单次transfer最多45秒，SSH连接和读写也有超时。长传输超时后保留partial文件，后续接着传；不会保证45秒内拉完任意大小的文件。
3. 截止前5分钟写出本地 `STOP_STARTING_GPU_JOBS`。实验控制器必须检查它，停止新开长任务，并为当前任务结束、报告拉取和校验留出时间。脚本不会擅自杀GPU进程。
4. 每个run结束，立即核对本地镜像里的 `job_result.json`、完整rank报告和source manifest；原始数据归档后再做独立审计。可用 `--checksum-last` 在最后阶段增加一次内容校验，但大文件校验耗时必须预留。
5. assignment结束前确认最后一次同步成功；不要只确认GPU任务exit0。脚本若最后一次失败或从未成功，退出码为1。每次尝试的退出码、时长、剩余时间与rsync输出均留存。

默认只复制远端根下 `reports/`、`archives/`、`state_reference_banks/`、`artifacts/`。可多次给 `--path NAME` 精确指定其他顶层结果目录；首次拿到机器后应确认所有最终结果、原始日志、模型配置和编译身份均位于这些路径。源代码先在本地保存再rsync上传，venv无需备份。GPU控制命令的stdout/stderr也应始终重定向到本地日志。

镜像位于独立输出目录的 `mirror/`，未完成传输保留在 `.rsync-partial/`，远端修改同名文件时旧本地版本保存在 `history/`。没有 `--delete`、`--inplace` 或远端删除操作；远端被清空、SSH失败都不会删除已有本地结果。完整性审计不能直接把partial文件或尚在写入的镜像当作完成结果。备份也不可能保证保存截止前尚未生成的数据。

`--dry-run` 只打印参数，不接触网络或创建目录。当前脚本已做本地协议/命令/超时/清空场景测试；由于本轮assignment已结束，没有声称它已在这台远端机器做过新的真实传输测试。
