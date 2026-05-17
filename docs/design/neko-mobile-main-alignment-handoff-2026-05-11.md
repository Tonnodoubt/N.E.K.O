# `neko_mobile` -> `main` 后端对齐进度文档（2026-05-11）

> 这份文档是给“下次回来的自己”看的。
> 目标是把“本轮目标、已经做完的事、当前卡点、下次继续步骤”放在一个地方。
>
> 更新时间：`2026-05-11`

---

## 1. 本轮目标

当前处理的是 **后端仓库** `/Users/tongqianqiu/N.E.K.O.TONG`，不是 RN 仓库本身。

核心目标：

- 以 `main` 为基线。
- 只把 `neko_mobile` 里对手机真正必需的后端能力手工移植回来。
- 不做全量 merge。
- 重点保证：手机能连、会话能起、图片/相机能走、P2P/LAN 能跑。

这一轮中途出现了一个关键变量：

- **云服务器已经换了。**
- 所以旧的云注册地址和旧的 STUN 地址不能再继续写死。

---

## 2. 当前状态总览

### 2.1 分支

- 当前仓库：`/Users/tongqianqiu/N.E.K.O.TONG`
- 当前分支：`main-mobile-align-2026-05-11`
- 当前起点 commit：`581072d9`

### 2.2 当前总体判断

可以把现在的状态理解成：

- mobile-critical 的后端骨架已经基本移植完。
- 代码已经到了“可启动、可测”的状态。
- 但因为云服务器切换，**还没有完成最后一次真实网络环境验证**。
- 现在最缺的不是继续大改代码，而是把新服务器配置填进去，再跑一轮 launcher + 手机 smoke。

---

## 3. 已完成的事情

### 3.1 已移植的主要能力

- `launcher.py`
  - 已接入 `LAN Proxy` 子进程启动。
  - 已把 `LAN_PROXY_PORT` 纳入启动信息和运行时重载逻辑。

- `app/main_server.py`
  - 已新增 `/p2p-info`
  - 已新增 `/lanproxyqrcode`
  - 主服务会通过本机 LAN Proxy 反查这些信息。

- `lan_proxy.py`
  - 已引入 LAN Proxy 主体实现。
  - 已支持 `/health`、`/p2p-info`、`/lanproxyqrcode`
  - 已支持 UDP P2P server 启动。
  - 已接云注册客户端。
  - 已移除运行路径中的旧 FRP 依赖。

- `udp_server.py`
  - 已加入仓库，作为 UDP P2P 服务端。

- `stun_client.py`
  - 已加入仓库。
  - 已包含 NAT 类型检测逻辑。

- `cloud_registry_client.py`
  - 已加入仓库。
  - 用于设备信息注册 / 查询。

- `cloud-registry/aliyun-api/main.py`
  - 已加入云注册服务端样例实现。

- `main_logic/core.py`
  - 图片流发送时已补 `image_source`
  - 当前相机路径会显式传 `image_source="camera"`

- `main_logic/omni_realtime_client.py`
  - `stream_image(...)` 已支持 `image_source`
  - `camera` 输入不会再被非 camera 的 idle multiplier 逻辑拖慢

- `main_logic/omni_offline_client.py`
  - 已补 `stream_image(..., image_source=...)` 签名兼容

- `config/__init__.py`
  - 已加入 `LAN_PROXY_PORT`

### 3.2 这一轮额外补的“换云收口”

这是这次最重要的补丁：

- `lan_proxy.py`
  - STUN 地址不再写死旧 IP。
  - 现在优先读取：
    - `NEKO_STUN_SERVER`
    - `NEKO_STUN_PORT`
  - 如果没配 STUN host，会自动跟随：
    - `NEKO_CLOUD_REGISTRY_URL` 的 hostname
  - 如果连云注册也没配，才回退到公共 STUN：
    - `stun.l.google.com:19302`

- `cloud_registry_client.py`
  - 支持 `NEKO_CLOUD_REGISTRY_URL`
  - 同时兼容 `CLOUD_REGISTRY_URL`
  - 会自动去掉尾部 `/`

- 文档和模板已补：
  - `docs/config/environment-vars.md`
  - `docs/zh-CN/config/environment-vars.md`
  - `docker/env.template`

这些地方已经写清楚：

- `NEKO_LAN_PROXY_PORT`
- `NEKO_CLOUD_REGISTRY_URL`
- `NEKO_STUN_SERVER`
- `NEKO_STUN_PORT`

---

## 4. 已完成的验证

### 4.1 语法 / 单测

已通过：

- `uv run python -m py_compile config/__init__.py launcher.py app/main_server.py main_logic/core.py main_logic/omni_realtime_client.py main_logic/omni_offline_client.py cloud_registry_client.py stun_client.py lan_proxy.py cloud-registry/aliyun-api/main.py`
- `uv run python -m pytest tests/unit/test_video_session.py -q`
  - 结果：`5 passed`

### 4.2 本地启动验证

此前已经跑通过一次：

- `uv run python launcher.py`

当时结果：

- `launcher` 能正常启动
- 新增的 `LAN Proxy` 进程能拉起
- `/p2p-info` 有返回
- `/lanproxyqrcode` 有返回

当时唯一没过的是：

- 云注册还在连旧服务器
- 报错类似：`Cannot connect to host 47.117.174.64:8000`

这也是后来补“环境变量收口”的直接原因。

### 4.3 新 STUN 解析逻辑已做快速校验

已经验证过：

- 只设 `NEKO_CLOUD_REGISTRY_URL=http://new-cloud.example.com:8000`
  - STUN 会解析成 `('new-cloud.example.com', 3478, 'cloud-registry')`

- 显式设：
  - `NEKO_STUN_SERVER=stun.example.com`
  - `NEKO_STUN_PORT=12345`
  - 会解析成 `('stun.example.com', 12345, 'env')`

这说明“跟随云注册 host”这条回退逻辑是通的。

### 4.4 2026-05-12 继续验证结果

已确认 `.env` 中生效的 `NEKO_CLOUD_REGISTRY_URL` 已换成新云地址，不再是旧 IP。

重新清掉旧的 orphaned N.E.K.O 子进程后，跑过一次干净启动：

- `uv run python launcher.py`
- Memory / Main / LAN Proxy / Agent 均能从当前工作区启动。
- `GET http://127.0.0.1:48911/health` 返回 `ok`。
- `GET http://127.0.0.1:48911/p2p-info` 返回 LAN Proxy JSON。
- `GET http://127.0.0.1:48911/lanproxyqrcode` 返回 `200`。
- `GET http://127.0.0.1:48920/health` 返回 `ok`。
- `GET http://127.0.0.1:48920/p2p-info` 返回连接信息。

当前新的阻塞点在云侧连通性：

- STUN 使用 `NEKO_CLOUD_REGISTRY_URL` 的 hostname + `3478`，但请求超时。
- 云注册 `POST /api/register` 超时。
- 本机探测新云 `8000/tcp` 超时。
- 本机探测新云 `3478/udp` 超时。
- 本机探测 `GET /api/health` 超时。

因此本地后端代码已经从“可启动”推进到“本机 LAN 链路可用”；剩下不是继续改本地后端，而是确认新云服务器上的注册服务 / STUN 服务是否已部署并开放端口。

### 4.5 2026-05-12 决策：云侧能力先封存

当前目标改成：先把同一局域网下“手机端 <-> 电脑后端”的连接做好。

已把 LAN Proxy 改成默认 LAN-only：

- `NEKO_ENABLE_CLOUD_REGISTRY=false` 默认不注册云端。
- `NEKO_ENABLE_STUN=false` 默认不跑 STUN 探测。
- 即使 `.env` 里保留 `NEKO_CLOUD_REGISTRY_URL`，默认也不会访问云服务器。
- `/p2p-info` 和 `/lanproxyqrcode` 继续返回 LAN 连接信息，供手机同 Wi-Fi 扫码连接。

以后如果要恢复跨网络 / P2P，再显式打开：

```dotenv
NEKO_ENABLE_CLOUD_REGISTRY=true
NEKO_ENABLE_STUN=true
NEKO_CLOUD_REGISTRY_URL=http://<cloud-host>:8000
# NEKO_STUN_SERVER=<stun-host>
# NEKO_STUN_PORT=3478
```

---

## 5. 当前未完成事项

### 5.1 必做

1. 重新跑：
   - `uv run python launcher.py`
2. 确认本机接口：
   - `GET http://127.0.0.1:48911/p2p-info`
   - `GET http://127.0.0.1:48911/lanproxyqrcode`
   - `GET http://127.0.0.1:48920/health`
3. 再做一轮同 Wi-Fi 手机真机 smoke：
   - 手机扫码拿到 LAN 连接信息
   - 文本
   - 语音
   - 相机图片
   - 角色切换

### 5.2 还没验证，不代表有 bug，但还没闭环

- 手机端是否能在同 Wi-Fi 真实环境中走完 QR / LAN 连接
- 这批后端改动还没有提交 commit
- 跨网络 / 云注册 / STUN 先封存，暂不作为本轮验收项

---

## 6. 下次继续就按这个顺序做

按这个顺序做最快：

### Step 1. 改环境变量

编辑仓库根目录 `.env`，至少改这个：

```dotenv
NEKO_CLOUD_REGISTRY_URL=http://<new-cloud-host>:8000
```

如果 STUN 不在同一台机器，再加：

```dotenv
NEKO_STUN_SERVER=<your-stun-host>
NEKO_STUN_PORT=3478
```

2026-05-12 已确认生效的云注册地址不是旧 IP；如果后续仍超时，优先排查云服务器服务进程、监听端口、安全组和防火墙。

### Step 2. 启动后端

```bash
cd /Users/tongqianqiu/N.E.K.O.TONG
uv run python launcher.py
```

预期观察点：

- `LAN Proxy` 正常启动
- STUN 不再出现旧 IP
- 云注册不再连 `47.117.174.64`
- 最好能看到注册成功日志

### Step 3. 本机接口自检

另开终端测：

```bash
curl -sf http://127.0.0.1:48920/p2p-info
curl -I http://127.0.0.1:48920/lanproxyqrcode
curl -sf http://127.0.0.1:48911/health
```

### Step 4. 手机真机 smoke

至少验证：

- 手机能拿到二维码或连接信息
- 能发文本
- 能起语音会话
- 相机图片不会明显卡死
- 角色切换不会把主流程打挂

---

## 7. 当前 git 状态

### 7.1 已 staged 的主改动

这些已经在 index 里：

- `app/main_server.py`
- `cloud-registry/aliyun-api/main.py`
- `cloud_registry_client.py`
- `config/__init__.py`
- `lan_proxy.py`
- `launcher.py`
- `main_logic/core.py`
- `main_logic/omni_offline_client.py`
- `main_logic/omni_realtime_client.py`
- `stun_client.py`
- `udp_server.py`

### 7.2 还没重新 add 的补充改动

这些文件后来又继续改了，所以现在是“已 staged 旧版本 + 工作区有新改动”：

- `cloud_registry_client.py`
- `lan_proxy.py`
- `stun_client.py`

另外还有这几个纯文档 / 模板改动还没 add：

- `docker/env.template`
- `docs/config/environment-vars.md`
- `docs/zh-CN/config/environment-vars.md`

### 7.3 与本轮无关的脏文件

这些不属于这次后端对齐主任务，之前就存在，暂时不要顺手处理：

- `baseline-smoke.json`
- `readme-smoke.json`
- `step*.json`
- `plugin/plugins/mahjong_companion/`
- `plugin/tests/data/`

---

## 8. 如果准备提交

先把这轮补充重新 add：

```bash
git add \
  cloud_registry_client.py \
  lan_proxy.py \
  stun_client.py \
  docker/env.template \
  docs/config/environment-vars.md \
  docs/zh-CN/config/environment-vars.md \
  docs/design/neko-mobile-main-alignment-handoff-2026-05-11.md
```

然后再看一次状态：

```bash
git status --short
```

如果确认没问题，再提交。

---

## 9. 一句话结论

一句话总结当前阶段：

- **代码层面的后端对齐已经差不多了。**
- **现在真正缺的是“换到新云服务器后再跑一次真环境验证”。**
- **下一步不要再无脑继续改代码，先填新服务器配置并实测。**

如果下次回来只记住一件事，就记这个：

> 先改 `.env` 里的新云地址，再跑 `uv run python launcher.py`，然后拿手机实测。

---

## 10. 相关文件

- 当前计划草稿（注意：在 `docs/plans/`，默认被 `.gitignore` 忽略）：
  - `docs/plans/neko-mobile-main-alignment-2026-05-11.md`

- 本轮重点文件：
  - `launcher.py`
  - `app/main_server.py`
  - `lan_proxy.py`
  - `udp_server.py`
  - `stun_client.py`
  - `cloud_registry_client.py`
  - `main_logic/core.py`
  - `main_logic/omni_realtime_client.py`
  - `main_logic/omni_offline_client.py`
  - `config/__init__.py`
