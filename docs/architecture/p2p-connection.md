# N.E.K.O 两层连接架构分析

## 概述

N.E.K.O 的移动端（RN）与桌面端（Python 后端）之间采用两层回退连接策略，从最优先的局域网直连，降级到 UDP 打洞。两层之间是串行尝试，前一层失败才进入下一层。

核心思路：**用 UDP 握手来发现 TCP endpoint，最终所有业务流量都走 HTTP/WebSocket（TCP）**。UDP 只是"敲门"用的，不承载实际数据。

如果两层都失败（如对称型 NAT、企业防火墙等场景），用户可自行配置第三方隧道工具（如 FRP、Tailscale、Cloudflare Tunnel 等）实现远程访问。

---

## 端口速查

| 角色 | 端口 | 说明 |
|------|------|------|
| main_server | 48911 | 核心业务服务（FastAPI） |
| lan_proxy (HTTP) | 48920 | LAN 代理，对外暴露的 HTTP 入口 |
| UDP P2P (STUN层) | 动态 | STUN 返回的公网端口，与 lan_proxy 绑定 |
| 云注册中心 | — | 阿里云 API，存储设备连接信息 |

---

## 第1层：LAN 直连

### 信息通道路线

```
[手机 RN App]
    │
    │  HTTP GET /health（5秒超时）
    ▼
[局域网 IP:48920]  ← lan_proxy.py 监听
    │
    │  反向代理
    ▼
[127.0.0.1:48911]  ← main_server.py（真正的业务服务）
```

### 核心逻辑

RN 端在启动时，从云注册中心拉取到桌面端的 `lanIp`（局域网 IP）和 `lanPort`（48920）。

直接用 HTTP `fetch` 请求 `/health` 接口，5 秒内收到 200 响应就认为 LAN 可达，直接把 `host:port` 设为这个地址，后续所有 API 请求都走这条路。

**成功条件**：手机和电脑在同一个 WiFi 下，且防火墙没有拦截 48920 端口。

**失败场景**：不在同一局域网，或者局域网内有隔离（如酒店 WiFi、企业网络）。

---

## 第2层：UDP STUN 打洞

### 信息通道路线（完整流程）

```
启动阶段（桌面端）：
[lan_proxy.py]
    │  向 STUN 服务器查询自己的公网地址
    ▼
[47.117.174.64:3478]  ← STUN 服务器
    │  返回：公网 IP + 公网端口（NAT 映射后的地址）
    ▼
[lan_proxy.py] 保存 stun_ip, stun_port
    │  注册到云端
    ▼
[云注册中心]  存储 {deviceId, lanIp, stunIp, stunPort, ...}


连接阶段（手机端发起）：
[手机 RN App]
    │  从云端拉取 stunIp:stunPort
    │
    │  UDP 发送 HELLO {type, token, timestamp}
    ▼
[公网 stunIp:stunPort]  ← NAT 映射到桌面端的 UDP socket
    ▼
[lan_proxy.py 内的 UDPP2PServer]
    │  验证 token
    │  回复 ACK {type, tcp_endpoint: {ip, port}}
    ▼
[手机 RN App]
    │  收到 ACK，提取 tcp_endpoint
    │  将 config.host/port 更新为 tcp_endpoint
    │
    │  后续所有业务请求走 HTTP/WebSocket
    ▼
[tcp_endpoint（即 stunIp:48920）]
    ▼
[lan_proxy.py HTTP 代理]
    ▼
[127.0.0.1:48911 main_server]
```

### 核心逻辑

STUN 的作用是让桌面端知道自己被 NAT 映射后的公网地址。桌面端启动时主动查询，把结果（stunIp:stunPort）注册到云端。

手机端拿到这个地址后，直接往上面发 UDP 包。如果 NAT 类型允许（Full Cone 或 Port Restricted Cone），这个 UDP 包能穿透 NAT 到达桌面端的 UDP socket。桌面端收到后，回复一个 ACK，里面带上 TCP 的访问地址（stunIp:48920）。

手机拿到 TCP 地址后，UDP 的使命就结束了，后续全部切换到 HTTP 走业务。

**关键点**：这里的"打洞"是单向的——手机往桌面端打，桌面端被动等待。严格意义上不是双向 UDP 打洞，更像是"UDP 探测 + TCP 回落"。对称型 NAT 下可能失败。

**失败场景**：对称型 NAT（如运营商级 NAT）、UDP 被防火墙拦截、STUN 服务器不可达。

---

## 两层对比

| 维度 | 第1层 LAN | 第2层 STUN |
|------|-----------|------------|
| 握手协议 | HTTP | UDP |
| 业务协议 | HTTP/WS | HTTP/WS |
| 依赖条件 | 同局域网 | NAT 类型友好 |
| 延迟 | 最低 | 中 |
| 稳定性 | 高 | 中（NAT 依赖） |
| 超时设置 | 5 秒 | 10 秒 |
| 中转节点 | 无 | 无（直连） |

> **注意**：如果两层均失败，用户可自行配置 FRP、Tailscale、Cloudflare Tunnel 等第三方隧道工具实现远程访问。

---

## 已知问题

**1. 事件监听时序错误（RN 端）**

`useUdpP2PConnection.ts` 里，`client.on('connected', ...)` 的注册发生在 `await client.connect()` 之后。此时 `connected` 事件早已 emit，监听永远不会触发，导致 UI 上的"连接层级"永远显示未知。

**2. ACK 中 tcp_ip 回退逻辑错误（后端）**

`udp_server.py` 的 `_handle_hello` 里：
```python
tcp_ip = self.tcp_ip or addr[0]
```
`addr[0]` 是客户端（手机）的 IP，不是服务器自身的地址。当 `tcp_ip` 未配置时，会把手机的 IP 当作 TCP endpoint 返回给手机，导致连接失败。

**3. STUN 层不是真正的双向打洞**

目前只有手机主动往桌面端发 UDP，桌面端被动等待。真正的 UDP 打洞需要双方同时互发，才能在对称型 NAT 下打通。当前实现在对称型 NAT 环境下会失败。

---

## 云注册中心的角色

云注册中心（阿里云 API）是整个流程的"地址簿"。桌面端启动后把自己的所有连接信息（lanIp、stunIp:stunPort、token）注册上去。手机端扫码或首次连接时从云端拉取这份信息，之后本地缓存到 AsyncStorage。

每次 RN 端发起连接前，如果有 `deviceId`，会先从云端刷新一次配置，确保拿到最新的 STUN 地址（因为 NAT 映射的公网端口可能会变）。
