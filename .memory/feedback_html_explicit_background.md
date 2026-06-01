---
name: feedback-html-explicit-background
description: HTML 文档必须显式设 body 的 background-color 和 color，否则 VS Code 暗色背景下读不到文字
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 617b2b7f-3773-47d2-a836-3744e7318bcc
---

写 HTML 文档（含 `<style>`）时，必须在 `body` 上显式设置 `background-color` 和 `color`，不能依赖浏览器/VS Code 的默认值。

**Why**：用户在 VS Code 里预览 HTML — VS Code 用暗色主题。如果 HTML 没显式设背景，浏览器/VS Code 渲染时背景可能继承暗色，但我的 `color: #1f2328`（深灰）几乎黑色，结果黑字配暗色背景完全看不清。用户实际遇到过这个问题（`doc/robocasa_gr00t_checkpoints.html` 第一版没设 background）。

**How to apply**：每次写带样式的 HTML 文档（CLAUDE.md 偏好的 single-file HTML+SVG 格式），`body` 选择器至少包含：
```css
body {
  background: #ffffff;   /* 或具体浅色，强制白底 */
  color: #1f2328;
  /* ... 其它样式 */
}
```
或者用 CSS 变量同时支持 light/dark，但**必须设默认值**。相关：[[architecture-doc-conventions]] HTML/SVG 文档规范。
