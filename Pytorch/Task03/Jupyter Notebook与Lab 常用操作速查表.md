---
文档名-title: Jupyter Notebook与Lab 常用操作速查表
创建时间-create time: 2025-07-17 22:37
更新时间-modefived time: 2025-07-17 22:37 星期四
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

## 上传多个文件的方法：

将文件打包成一个zip压缩包
上传该压缩包
解压文件!unzip (压缩包所在路径) -d (解压路径)，例如：!unzip coco.zip -d data/coco
删除该压缩包

### 常用快捷键[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id7 "永久链接至标题")

1. 入门操作
    
    # 增加，减少，剪切，保存，删除等
    # a, b, x, s, dd
    
    # 合并，执行本单元代码，并跳转到下一单元，执行本单元代码，留在本单元
    # Shift+M Shift+Enter Ctrl+Enter
    
    # 显示行数，切换markdown/code
    # l, m/y
    
2. Jupyter Notebook中按下Enter进入编辑模式，按下Esc进入命令模式
    

- **编辑模式（绿色）**
    
    ![image-20220805144702521](https://datawhalechina.github.io/thorough-pytorch/_images/image-20220805144702521.png)
    
- **命令模式（蓝色）**
    
    ![image-20220805144722375](https://datawhalechina.github.io/thorough-pytorch/_images/image-20220805144722375.png)
    
- 在命令模式下，点击h，会弹出快捷键窗口
    

![image-20220805144803441](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220805144803441.png)

1. Jupyter Lab同样有两种模式。按下Enter进入编辑模式，按下Esc进入命令模式
    

- 编辑模式（有框线无光标）
    

![image-20220805145218241](https://datawhalechina.github.io/thorough-pytorch/_images/image-20220805145218241.png)

- 命令模式（无框线无光标）
    

![image-20220805150721875](https://datawhalechina.github.io/thorough-pytorch/_images/image-20220805150721875.png)

## 快捷键汇总
    

​ 命令模式（按`Esc`）

- **Enter** : 转入编辑模式
    
- **Shift-Enter** : 运行本单元，选中下个单元
    
- **Ctrl-Enter** : 运行本单元
    
- **Alt-Enter** : 运行本单元，在其下插入新单元
    
- **Y** : 单元转入代码状态
    
- **M** :单元转入markdown状态
    
- **R** : 单元转入raw状态
    
- **1** : 设定 1 级标题
    
- **2** : 设定 2 级标题
    
- **3** : 设定 3 级标题
    
- **4** : 设定 4 级标题
    
- **5** : 设定 5 级标题
    
- **6** : 设定 6 级标题
    
- **Up** : 选中上方单元
    
- **K** : 选中上方单元
    
- **Down** : 选中下方单元
    
- **J** : 选中下方单元
    
- **Shift-K** : 扩大选中上方单元
    
- **Shift-J** : 扩大选中下方单元
    
- **A** : 在上方插入新单元
    
- **B** : 在下方插入新单元
    
- **X** : 剪切选中的单元
    
- **C** : 复制选中的单元
    
- **Shift-V** : 粘贴到上方单元
    
- **V** : 粘贴到下方单元
    
- **Z** : 恢复删除的最后一个单元
    
- **D,D** : 删除选中的单元
    
- **Shift-M** : 合并选中的单元
    
- **Ctrl-S** : 文件存盘
    
- **S** : 文件存盘
    
- **L** : 转换行号
    
- **O** : 转换输出
    
- **Shift-O** : 转换输出滚动
    
- **Esc** : 关闭页面
    
- **Q** : 关闭页面
    
- **H** : 显示快捷键帮助
    
- **I,I** : 中断Notebook内核
    
- **0,0** : 重启Notebook内核
    
- **Shift** : 忽略
    
- **Shift-Space** : 向上滚动
    
- **Space** : 向下滚动
    

​ 编辑模式（按`Enter`）

- **Tab** : 代码补全或缩进
    
- **Shift-Tab** : 提示
    
- **Ctrl-]** : 缩进
    
- **Ctrl-[** : 解除缩进
    
- **Ctrl-A** : 全选
    
- **Ctrl-Z** : 复原
    
- **Ctrl-Shift-Z** : 再做
    
- **Ctrl-Y** : 再做
    
- **Ctrl-Home** : 跳到单元开头
    
- **Ctrl-Up** : 跳到单元开头
    
- **Ctrl-End** : 跳到单元末尾
    
- **Ctrl-Down** : 跳到单元末尾
    
- **Ctrl-Left** : 跳到左边一个字首
    
- **Ctrl-Right** : 跳到右边一个字首
    
- **Ctrl-Backspace** : 删除前面一个字
    
- **Ctrl-Delete** : 删除后面一个字
    
- **Esc** : 进入命令模式
    
- **Ctrl-M** : 进入命令模式
    
- **Shift-Enter** : 运行本单元，选中下一单元
    
- **Ctrl-Enter** : 运行本单元
    
- **Alt-Enter** : 运行本单元，在下面插入一单元
    
- **Ctrl-Shift--** : 分割单元
    
- **Ctrl-Shift-Subtract** : 分割单元
    
- **Ctrl-S** : 文件存盘
    
- **Shift** : 忽略
    
- **Up** : 光标上移或转入上一单元
    
- **Down** :光标下移或转入下一单元
    

### 3.3 安装插件[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id8 "永久链接至标题")

- Jupyter Notebook安装插件的方法
    

1. 在Anaconda Powershell Prompt中输入
    
    pip install jupyter_contrib_nbextensions
    
2. 再次输入以下指令，将插件添加到工具栏
    
    jupyter contrib nbextension install
    
3. 打开Jupyter Notebook，点击Nbextensions，取消勾选`disable configuration for nbextensions without explicit compatibility`，此时可以添加自己喜欢的插件啦！
    

![image-20220805151600140](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220805151600140.png)

1. 推荐以下两个基础插件
    

- Execute Time：可以显示执行一个Cell要花费多少时间
    
- Hinterland：提供代码补全功能
    
- Jupyter Lab安装插件的方法
    

1. Jupyter Lab安装插件点击左侧的第四个标志，点击“Enable”后就可以在搜索栏中搜索想要的插件
    

![image-20220805152331777](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220805152331777.png)

1. 例如搜索`jupyterlab-execute-time`后，在Search Results中查看结果，点击Install便可安装插件
    

![image-20220805152507891](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220805152507891.png)

1. 还可以在Anaconda Powershell Prompt中使用指令来安装插件
    
    jupyter labextension install jupyterlab-execute-time  # 安装jupyterlab-execute-time