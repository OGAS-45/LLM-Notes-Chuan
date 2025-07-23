---
文档名-title: Dify课程笔记章节08 - dify联通mysql 和 echart
创建时间-create time: 2025-07-23 21:26
更新时间-modefived time: 2025-07-23 21:26 星期三
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---


## 前一半都是针对数据库mysql的，我就不用学了，早就会了

## 后半段：3. # dify使用mysql 及 echart


教程中很有部分的内容：

标准的Agent数据库查询Prompt
```SQL
表结构设计 ：
- comments ：存储评论ID、查询内容和情绪分类
- keywords ：存储唯一关键字及自增ID（确保关键字不重复）
- comment_keywords ：多对多关联表，连接评论和关键字
   
三个表的基本信息
1. comments表（评论表）
字段名
数据类型
说明
约束
comment_id
INT
评论ID（主键）
PRIMARY KEY
query
TEXT
评论内容
NOT NULL
sentiment
VARCHAR(20)
情绪分类（积极/消极/中性）
NOT NULL

2. keywords表（关键字表）
字段名
数据类型
说明
约束
keyword_id
INT
关键字ID（主键）
PRIMARY KEY
keyword
VARCHAR(255)
关键字内容
NOT NULL, UNIQUE

3. comment_keywords表（评论-关键字关联表）
字段名
数据类型
说明
约束
comment_id
INT
评论ID（外键）
FOREIGN KEY
keyword_id
INT
关键字ID（外键）
FOREIGN KEY

复合主键
PRIMARY KEY (comment_id, keyword_id)

表关联关系
- 一对多关系：一个评论可以对应多个关键字（通过comment_keywords表）
- 一对多关系：一个关键字可以属于多个评论（通过comment_keywords表）
- 多对多关系：comments表和keywords表通过comment_keywords表建立多对多关联
  
基本查询SQL案例

1. 查询所有积极情绪的评论及其关键字
SELECT 
  c.comment_id, 
  c.query AS comment_content, 
  GROUP_CONCAT(k.keyword SEPARATOR ', ') AS keywords
FROM 
  comments c
JOIN 
  comment_keywords ck ON c.comment_id = ck.comment_id
JOIN 
  keywords k ON ck.keyword_id = k.keyword_id
WHERE 
  c.sentiment = '积极'
GROUP BY 
  c.comment_id, c.query;

2. 统计各情绪类型的评论数量
SELECT 
  sentiment, 
  COUNT(*) AS comment_count
FROM 
  comments
GROUP BY 
  sentiment;

3. 查询出现频率最高的前10个关键字
SELECT 
  k.keyword, 
  COUNT(ck.keyword_id) AS occurrence_count
FROM 
  keywords k
JOIN 
  comment_keywords ck ON k.keyword_id = ck.keyword_id
GROUP BY 
  k.keyword
ORDER BY 
  occurrence_count DESC
LIMIT 10;

4. 查询包含特定关键字的所有评论
SELECT 
  c.comment_id, 
  c.query AS comment_content, 
  c.sentiment
FROM 
  comments c
JOIN 
  comment_keywords ck ON c.comment_id = ck.comment_id
JOIN 
  keywords k ON ck.keyword_id = k.keyword_id
WHERE 
  k.keyword = '环保';
```


抽取data真的很好用
```Python
用户输入内容，将json中的y/x data值抽取出来。注意只输出抽取的数据结果。结果用分号分割。

例如输入的：
{
"xdata": ["消极"; "中性"; "积极"],
"ydata": [121; 133; 95]
}}

输出：
121; 133; 95

输入：
{agent/text}


输出：
```

最牛的是最后的结合数据，其实prompt并不长，让其自由发挥

```
你是一个数据分析师，请根据

{{#sys.query#}}进行分析。

  

通过sql查询后得到的分析结果：
{{#1753277478162.text#}}
  

目前有{{#1753277450007.output#}}数据表
```

![[Pasted image 20250723214914.png]]十分有用的查询手册