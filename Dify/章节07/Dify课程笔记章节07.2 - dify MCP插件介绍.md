---
文档名-title: Dify课程笔记章节07.2 - dify MCP插件介绍
创建时间-create time: 2025-07-19 22:20
更新时间-modefived time: 2025-07-19 22:20 星期六
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

# dify MCP插件介绍

## 2.1 dify 插件介绍

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/505aac3d-536a-4a92-a2bc-87ba00f9fda5.png)

在 v1.0.0 之前，Dify 平台面临一个关键挑战： **模型和工具与主平台高度耦合，新增功能需要修改主仓库代码，限制了开发效率和创新** 。为此，Dify团队重构了 Dify 底层架构，引入了全新的插件机制，带来了以下四大优势：

- **组件插件化：** 插件与主平台解耦，模型和工具以插件形式独立运行，支持单独更新与升级。新模型的适配不再依赖于 Dify 平台的整体版本升级，用户只需单独更新相关插件，无需担心系统维护和兼容性问题。新工具的开发和分享将更加高效，支持接入各类成熟的软件解决方案和工具创新。
    
- **开发者友好：** 插件遵循统一的开发规范和接口标准，配备远程调试、代码示例和 API 文档的工具链，帮助插件开发者快速上手。
    
- **热插拔设计：** 支持插件的动态扩展与灵活使用，确保系统高效运行。
    
- **多种分发机制：**
    

**Dify Marketplace：** 作为插件聚合、分发与管理平台，为所有 Dify 用户提供丰富的插件选择。插件开发者可将开发好的插件包提交至 Dify Plugins 仓库，通过 Dify 官方的代码和隐私政策审核后即可上架 Marketplace。Dify Marketplace 现共有 120+ 个插件，其中包括：

**模型：** OpenAI o1 系列（o1、o3-mini 等）、Gemini 2.0 系列、DeepSeek-R1 及其供应商，包括硅基流动、OpenRouter、Ollama、Azure AI Foundry、Nvidia Catalog 等。 **工具：** Perplexity、Discord、Slack、Firecrawl、Jina AI、Stability、ComfyUI、Telegraph 等。更多插件尽在 Dify Marketplace。请通过插件帮助文档查看如何将开发好的插件发布至 Marketplace。

> Dify 插件帮助文档 >> https://docs.dify.ai/zh-hans/plugins/introduction

**社区共享：** 通过 GitHub 等在线社区，插件开发者可以自由分享插件，促进开源合作与社区创新。

**本地私域：** 社区版和企业版用户可以在本地部署和管理插件，沉淀个人和组织内部的 AI 应用开发工具和解决方案，加速 AI 应用的落地，促进团队资源共享。

![[Pasted image 20250719222353.png]]

![[Pasted image 20250719222400.png]]

![[Pasted image 20250719222407.png]]

![[Pasted image 20250719222415.png]]

![[Pasted image 20250719222513.png]]

![[Pasted image 20250719222517.png]]

在dify的丰富插件市场中也提供了一个好用的MCP SSE插件，方便我们将SSE MCP服务放在我们的工作流中。让AI拥有更加强大的能力。

2.3 下载 MCP SSE / StreamableHTTP 插件

参加了豆包大模型中的调用MCP，很有意思。