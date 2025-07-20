---
文档名-title: Dify课程笔记章节07 - MCP综合应用理论学习
创建时间-create time: 2025-07-19 22:07
更新时间-modefived time: 2025-07-19 22:07 星期六
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

# Agent = 插上腿的LLM 

大语言模型，例如DeepSeek，如果不能联网、不能操作外部工具，只能是聊天机器人。除了聊天没什么可做的。

而一旦大语言模型能操作工具，例如：联网/地图/查天气/函数/插件/API接口/代码解释器/机械臂/灵巧手，它就升级成为智能体Agent，能更好地帮助人类。今年爆火的Manus就是这样的智能体。

众多大佬、创业公司，都在All In押注AI智能体赛道。

# 以前的Agent怎么实现 —— 大段Prompt

```
比如在openai中这是一个用于处理客户订单配送日期查询的工具调用逻辑设计。以下是关键点解读：

### 一、工具功能解析

1. 核心用途
    
    - 函数名 `get_delivery_date` 明确用于查询订单的配送日期​（预计送达时间）。
        
    - 触发场景：当用户询问包裹状态（如“我的包裹到哪里了？”或“预计何时送达？”）时自动调用。
        
2. 参数设计
    
    - 必需参数​：仅需提供 `order_id` （字符串类型），无需其他字段。
        
    - 逻辑合理性：订单ID是唯一标识，足以关联物流信息（如快递单号、配送进度等）。
        
3. 技术实现要求
    
    - 开发者需在后端实现该函数，通过 `order_id` 关联数据库或物流API获取实时配送状态（如预计送达时间、当前物流节点等）。
        

---

### 二、客服对话流程示例

假设用户提问：​​“Hi, can you tell me the delivery date for my order?”​​

助手应执行以下步骤：

1. 识别意图​：用户明确要求“delivery date”，符合工具调用条件。
    
2. 参数提取​：需引导用户提供 `order_id` （因消息中未直接包含该信息）：
    

> _“Sure! Please provide your order ID so I can check the delivery schedule.”_

1. 工具调用​：获得 `order_id` 后，后台执行 `get_delivery_date(order_id="XXX")` 。
    
2. 返回结果​：向用户展示函数返回的配送日期（如 _“您的订单预计在2025年6月25日18:00前送达”_ ）。
    

```
```json
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
    {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"}
]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
```

问题：

- 面对相同的函数和工具，每个开发者都需要 **重新从头造轮子** ，按照自己想要的模型回复格式重新撰写、调试提示词

- 百度地图发布的大模型工具调用接口，和高德地图发布接口，可能完全不一样。
- **没有统一的市场和生态** ，只能各自为战，各自找开发者接各自的大模型。

# MCP 秦王扫六合

Anthropic公司（就是发布Claude大模型的公司），在2024年11月，发布了Model Context Protocol协议，简称MCP

Type-C扩展坞， **让海量的软件和工具，能够插在大语言模型上，供大模型调用**

## 几个MCP的应用案例

调用Unity的MCP接口，让AI自己开发游戏。https://www.bilibili.com/video/BV1kzoWYXECJ

调用Blender的MCP接口，让AI自己3D建模。https://www.bilibili.com/video/BV1pHQNYREAX

调用百度地图的MCP接口，让AI自己联网，查路况，导航。https://www.bilibili.com/video/BV1dbdxY5EUP

调用Playwright的MCP接口，让AI自己操作网页。（后面的保姆级教程讲的就是这个）

## MCP就是腿

解决的核心问题：统一了大模型调用工具的方法

不同的浏览器，用相同的HTTP协议，就可以访问海量的网站。
不同的大模型，用相同的MCP协议，就可以调用海量的外部工具。
互联网催生出搜索、社交、外卖、打车、导航、外卖等无数巨头。
MCP同样可能催生出繁荣的智能体生态。
**类比互联网的HTTP协议，所有的智能体都值得用MCP重新做一遍。**

# MCP协议的通信双方

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/7e1b1004-db53-4846-9094-d05c8e427dab.png)

**MCP Host** ：人类电脑上安装的客户端软件，一般是Dify、Cursor、Claude Desktop、Cherry Studio、Cline，软件里带了大语言模型，后面的教程会带你安装配置。

**MCP Server** ：各种软件和工具的MCP接口，比如： 百度地图、高德地图、游戏开发软件Unity、三维建模软件Blender、浏览器爬虫软件Playwrights、聊天软件Slack。尽管不同软件有不同的功能，但都是以MCP规范写成的server文件，大模型一眼就知道有哪些工具，每个工具是什么含义。

有一些MCP Server是可以联网的，比如百度地图、高德地图。而有一些MCP Server只进行本地操作，比如Unity游戏开发、Blender三维建模、Playwright浏览器操作。

## MCP的Host、Client、Server是什么关系？

Host就是Dify、Cursor、Cline、CherryStudio等MCP客户端软件。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/176f6769-245a-4038-8d5d-040d0575190e.png)

如果你同时配置了多个MCP服务，比如百度地图、Unity、Blender等。每个MCP服务需要对应Host中的一个Client来一对一通信。Client被包含在Host中。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/bf078e72-f8fe-4063-8ad3-5106a7d87536.png)

## 大模型是怎么知道有哪些工具可以调用，每个工具是做什么的？

每个支持MCP的软件，都有一个MCP Server文件，里面列出了所有支持调用的函数，函数注释里的内容是给AI看的，告诉AI这个函数是做什么用的。

**MCP Server文件就是给AI看的工具说明书。**

例如百度地图MCP案例：

https://github.com/baidu-maps/mcp/blob/main/src/baidu-map/python/src/mcp_server_baidu_maps/map.py

每个以 `@mcp.tool()` 开头的函数，都是一个百度地图支持MCP调用的功能。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/a3975f0e-4d51-472f-be8a-316c5cee02d2.png)

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/59dc010e-c7f0-40be-9bdf-f589101e757d.png)

你也可以按照这个规范，自己开发MCP Server，让你自己的软件支持MCP协议，让AI能调用你软件中的功能。

## 参考资料

几张图片来自公众号：西二旗生活指北

教程引用来源：[MCP是什么？](https://zihao-ai.feishu.cn/wiki/RlrhwgNqLiW7VYkNnvscHxZjngh)

[MCP的技术细节（看不懂可跳过）](https://zihao-ai.feishu.cn/wiki/WhQlwydkMieX4Tki7lbcHK6TnUe)

官方介绍：https://mp.weixin.qq.com/s/CDhqmLO1JXSB__aUMqoGoQ