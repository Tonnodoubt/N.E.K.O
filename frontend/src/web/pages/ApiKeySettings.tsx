/**
 * API Key Settings Page
 *
 * Migrated from templates/api_key_settings.html
 * Uses neko theme system from theme.css
 * Now connected to real backend API
 */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import type { ChangeEvent } from "react";
import QRCode from "qrcode";
import "./ApiKeySettings.css";
import { getCoreConfig, updateCoreConfig } from "../api/config";

type ApiProvider = "free" | "ali" | "glm" | "step" | "silicon" | "openai" | "gemini";

interface ApiConfig {
  coreApiProvider: ApiProvider;
  coreApiKey: string;
  assistApiProvider: ApiProvider;
  assistApiKeys: {
    ali: string;
    openai: string;
    glm: string;
    step: string;
    silicon: string;
    gemini: string;
  };
  mcpRouterToken: string;
}

const API_PROVIDERS = [
  { value: "free", label: "免费版（推荐新手）" },
  { value: "ali", label: "阿里云百炼" },
  { value: "glm", label: "智谱 AI" },
  { value: "step", label: "阶跃星辰" },
  { value: "silicon", label: "硅基流动" },
  { value: "openai", label: "OpenAI" },
  { value: "gemini", label: "Gemini" },
];

// Map backend field names to frontend field names
const ASSIST_KEY_MAP: Record<string, keyof ApiConfig["assistApiKeys"]> = {
  qwen: "ali",
  openai: "openai",
  glm: "glm",
  step: "step",
  silicon: "silicon",
  gemini: "gemini",
};

export default function ApiKeySettings() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [p2pQrUrl, setP2pQrUrl] = useState<string | null>(null);
  const [p2pLoading, setP2pLoading] = useState(false);
  const [config, setConfig] = useState<ApiConfig>({
    coreApiProvider: "free",
    coreApiKey: "",
    assistApiProvider: "free",
    assistApiKeys: {
      ali: "",
      openai: "",
      glm: "",
      step: "",
      silicon: "",
      gemini: "",
    },
    mcpRouterToken: "",
  });

  // Load existing config
  useEffect(() => {
    loadConfig();
  }, []);

  // Load P2P info and generate QR code
  useEffect(() => {
    loadP2PInfo();
  }, []);

  const loadP2PInfo = async () => {
    try {
      setP2PLoading(true);

      // 获取 P2P 连接信息
      const apiBase = window.location.origin;
      const response = await fetch(`${apiBase}/p2p-info`);

      if (!response.ok) {
        console.log("P2P info not available");
        return;
      }

      const p2pInfo = await response.json();

      // 生成二维码
      const qrData = JSON.stringify(p2pInfo);
      const qrUrl = await QRCode.toDataURL(qrData, {
        width: 200,
        margin: 2,
        color: {
          dark: "#000000",
          light: "#ffffff",
        },
      });

      setP2pQrUrl(qrUrl);
    } catch (err) {
      console.log("Failed to load P2P info:", err);
    } finally {
      setP2PLoading(false);
    }
  };

  const loadConfig = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await getCoreConfig();

      // Map backend response to frontend state
      setConfig({
        coreApiProvider: (data.coreApi || "free") as ApiProvider,
        coreApiKey: data.api_key || "",
        assistApiProvider: (data.assistApi || "free") as ApiProvider,
        assistApiKeys: {
          ali: data.assistApiKeyQwen || "",
          openai: data.assistApiKeyOpenai || "",
          glm: data.assistApiKeyGlm || "",
          step: data.assistApiKeyStep || "",
          silicon: data.assistApiKeySilicon || "",
          gemini: data.assistApiKeyGemini || "",
        },
        mcpRouterToken: data.mcpToken || "",
      });
    } catch (err) {
      console.error("Failed to load config:", err);
      setError("加载配置失败，请刷新页面重试");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      // Map frontend state to backend request format
      const requestData = {
        coreApiKey: config.coreApiKey,
        coreApi: config.coreApiProvider,
        assistApi: config.assistApiProvider,
        assistApiKeyQwen: config.assistApiKeys.ali,
        assistApiKeyOpenai: config.assistApiKeys.openai,
        assistApiKeyGlm: config.assistApiKeys.glm,
        assistApiKeyStep: config.assistApiKeys.step,
        assistApiKeySilicon: config.assistApiKeys.silicon,
        assistApiKeyGemini: config.assistApiKeys.gemini,
        mcpToken: config.mcpRouterToken,
      };

      const result = await updateCoreConfig(requestData);

      if (result.success) {
        alert(result.message || "配置保存成功！");
      } else {
        setError(result.error || "保存失败，请重试");
      }
    } catch (err: any) {
      console.error("Failed to save config:", err);
      setError(err.message || "保存失败，请检查网络连接");
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    navigate("/");
  };

  if (loading) {
    return (
      <div className="neko-container">
        <div className="neko-loading">
          <div className="neko-loading-spinner"></div>
          <span className="neko-loading-text">正在加载配置...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="neko-container">
      {/* Header */}
      <div className="neko-header">
        <h2 data-text="API Key 设置">API Key 设置</h2>
        <button className="neko-close-btn" onClick={handleClose} title="关闭">
          <img src="/static/icons/close_button.png" alt="关闭" />
        </button>
      </div>

      <div className="neko-content">
        {/* Error Message */}
        {error && (
          <div className="neko-card neko-error-box">
            <p>❌ {error}</p>
            <button className="neko-btn neko-btn-secondary" onClick={() => setError(null)}>
              关闭
            </button>
          </div>
        )}

        {/* Quick Start Guide */}
        <section className="neko-info-box api-guide-box">
          <h3>🔑 快速开始</h3>
          <ul>
            <li>
              <strong>免费版（推荐新手）：</strong>
              <br />
              无需注册，在下方高级选项中选择"免费版"即可使用
            </li>
            <li>
              <strong>完整版（获取API Key）：</strong>
            </li>
            <li>
              <em>这只是一个例子，你完全可以使用OpenAI，Gemini或其他选项</em>
            </li>
            <li>
              1. 在阿里云注册账号并{" "}
              <a href="https://myaccount.console.aliyun.com/overview" target="_blank" rel="noopener noreferrer">
                完成实名认证
              </a>
            </li>
            <li>
              2. 访问阿里云百炼平台{" "}
              <a href="https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key" target="_blank" rel="noopener noreferrer">
                APIKey管理页面
              </a>
            </li>
            <li>3. 创建新的API Key并点击复制按钮</li>
            <li>4. 将API Key粘贴到下方输入框中</li>
          </ul>
        </section>

        {/* Newbie Recommendation */}
        <section className="neko-card api-recommend-box">
          <h3>✨ 新手推荐</h3>
          <p>
            如果您还没有API Key，可以直接在下方高级选项中选择"免费版"开始使用，无需注册任何账号！
          </p>
        </section>

        {/* Core API Configuration */}
        <section className="neko-card config-section">
          <h3>⚙️ 核心API配置</h3>
          <p className="neko-tips">请确保API Key格式正确。保存后需要重启服务才能生效。</p>

          <div className="neko-field-row">
            <label className="neko-label">
              核心API服务商
              <span className="tooltip" title="核心API负责对话功能">
                (?)
              </span>
            </label>
            <select
              className="neko-select"
              value={config.coreApiProvider}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                setConfig({ ...config, coreApiProvider: e.target.value as ApiProvider })
              }
            >
              {API_PROVIDERS.map((provider) => (
                <option key={provider.value} value={provider.value}>
                  {provider.label}
                </option>
              ))}
            </select>
          </div>

          <div className="neko-field-row">
            <label className="neko-label">
              核心API Key
              {config.coreApiProvider === "free" && (
                <span className="free-hint">（免费版无需填写）</span>
              )}
            </label>
            <input
              className="neko-input"
              type="text"
              value={config.coreApiKey}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setConfig({ ...config, coreApiKey: e.target.value })}
              placeholder="Enter your API Key"
              disabled={config.coreApiProvider === "free"}
            />
          </div>

          <button
            className="neko-btn neko-btn-primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "保存中..." : "💾 保存设置"}
          </button>
        </section>

        {/* Advanced Options - Fold Panel */}
        <section className={`neko-fold ${advancedOpen ? "open" : ""}`}>
          <button
            className="neko-fold-toggle"
            onClick={() => setAdvancedOpen(!advancedOpen)}
          >
            <span className={`fold-arrow ${advancedOpen ? "open" : ""}`}>▼</span>
            高级选项（可选）
          </button>

          <div className="neko-fold-content">
            <div className="neko-info-box advanced-info-box">
              <strong>配置建议：</strong>
              <br />
              • 免费版：完全免费，无需API Key，适合新手体验（不支持自定义语音、Agent模式和视频对话）
              <br />
              • 核心API：负责对话功能，建议根据预算和需求选择
              <br />
              • 辅助API：负责记忆管理和自定义语音，只有阿里支持自定义语音
            </div>

            {/* Assist API Configuration */}
            <div className="neko-field-row">
              <label className="neko-label">辅助API（记忆管理/自定义语音/文本聊天）</label>
              <select
                className="neko-select"
                value={config.assistApiProvider}
                onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                  setConfig({ ...config, assistApiProvider: e.target.value as ApiProvider })
                }
              >
                {API_PROVIDERS.map((provider) => (
                  <option key={provider.value} value={provider.value}>
                    {provider.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Assist API Keys for different providers */}
            {Object.entries(config.assistApiKeys).map(([provider, key]) => (
              <div key={provider} className="neko-field-row">
                <label className="neko-label">辅助API Key - {provider.toUpperCase()}</label>
                <input
                  className="neko-input"
                  type="text"
                  value={key}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    setConfig({
                      ...config,
                      assistApiKeys: {
                        ...config.assistApiKeys,
                        [provider]: e.target.value,
                      },
                    })
                  }
                  placeholder={`Optional, defaults to Core API Key`}
                />
              </div>
            ))}

            {/* MCP Router Token */}
            <div className="neko-field-row">
              <label className="neko-label">
                MCP Router Token
                <span className="tooltip" title="用于访问MCP Router服务的认证令牌">
                  (?)
                </span>
              </label>
              <input
                className="neko-input monospace"
                type="text"
                value={config.mcpRouterToken}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setConfig({ ...config, mcpRouterToken: e.target.value })
                }
                placeholder="Enter MCP Router Token (optional)"
              />
              <p className="neko-tips">
                当前版本请自行获取并配置{" "}
                <a href="https://mcp-router.net/" target="_blank" rel="noopener noreferrer">
                  MCP Router
                </a>
                ，后续版本会优化。
              </p>
            </div>

            <div className="neko-card coming-soon-box">
              <p>
                <strong>📌 自定义API配置</strong>
                <br />
                （摘要、纠错、情感、视觉、Agent、实时、TTS、GPT-SoVITS 等高级配置）
                <br />
                <em>功能开发中，敬请期待...</em>
              </p>
            </div>
          </div>
        </section>

        {/* P2P Connection QR Code */}
        {p2pQrUrl && (
          <section className="neko-card p2p-qr-section">
            <h3>📱 P2P 连接二维码</h3>
            <p className="neko-tips">使用手机 App 扫描二维码，即可在局域网内直连</p>

            <div className="p2p-qr-container">
              <img src={p2pQrUrl} alt="P2P Connection QR Code" className="p2p-qr-image" />
            </div>

            <div className="p2p-qr-info">
              <p className="neko-tips">
                ✅ 三层连接支持：LAN直连 → STUN打洞 → FRP中转
              </p>
            </div>
          </section>
        )}

        {p2pLoading && !p2pQrUrl && (
          <section className="neko-card p2p-qr-section">
            <div className="p2p-qr-loading">
              <div className="neko-loading-spinner"></div>
              <span>加载 P2P 连接信息...</span>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
