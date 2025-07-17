---
文档名-title: Jupyter Notebook
创建时间-create time: 2025-07-17 22:36
更新时间-modefived time: 2025-07-17 22:36 星期四
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---
## 1 Jupyter Notebook/Lab安装[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id1 "永久链接至标题")

1. 安装Jupyter Notebook：激活虚拟环境后，我们只需要在终端输入指令
    
    conda install jupyter notebook
    # pip install jupyter notebook
    

- 注：如果pip版本过低，还需提前运行更新pip的指令
    
    pip install --upgrade pip
    

1. 安装Jupyter Lab：激活环境后，我们同样也只需要在终端输入指令
    
    conda install -c conda-forge jupyterlab
    # pip install jupyterlab
    
2. 在终端输入指令打开Jupyter Notebook
    
    jupyter notebook  # 打开Jupyter Notebook
    jupyter lab  # 打开Jupyter Lab
    

- 如果浏览器没有自动打开Jupyter Notebook或者Jupyter Lab，复制端口信息粘贴至浏览器打开
    
    ![image-20220812162810305](https://datawhalechina.github.io/thorough-pytorch/_images/image-20220812162810305.png)
    
- 如果想要自定义端口，在终端输入如下指令修改
    
    jupyter notebook --port <port_number>
    
- 如果想启动服务器但不打开浏览器，可以在终端输入
    
    jupyter notebook --no-browser
    

## 2 Jupyter Notebook/Lab配置[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id2 "永久链接至标题")

### 2.1 设置文件存放位置[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id3 "永久链接至标题")

​ 在使用Jupyter Notebook/Jupyter Lab时，如果我们想要更改默认文件存放路径，该怎么办？

- Jupyter Notebook
    

1. 我们首先需要查看配置文件，只需要在终端输入
    
    jupyter notebook --generate-config
    
2. 我们记住出现配置文件的路径，复制到文件夹中打开（终端这里可以写N）
    

![image-20220716235510697](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220716235510697.png)

1. 在文件夹中双击打开配置文件
    

![image-20220716235540827](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220716235540827.png)

1. 打开Python文件后，用`Ctrl+F`快捷键查找，输入关键词，找到`# c.NotebookApp.notebook_dir = ''`
    
2. 去掉注释，并填充路径`c.NotebookApp.notebook_dir = 'D:\\Adatascience'`
    

![image-20220804170455298](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220804170455298.png)

1. 此时我们在终端中输入`jupyter notebook`，打开页面后发现文件默认路径已经被更改。但是点击菜单栏中的应用快捷方式打开Jupyter Notebook，打开页面发现文件位置仍然是默认路径
    

![image-20220716235931573](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220716235931573.png)

1. 如果我们想要更改应用快捷方式Jupyter Notebook的文件位置，此时需要右键选中快捷方式，打开文件所在位置。再右键点击快捷方式，查看属性，再点击快捷方式
    

![image-20220717000530228](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220717000530228.png) ![image-20220804171027983](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220804171027983.png)

1. 我们只需要在“目标”中删除红框标记部分，点击确定
    

![image-20220717001243904](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220717001243904.png)

1. 此时再打开菜单栏中Jupyter Notebook的快捷方式，发现页面文件路径已经变为之前自主设置的路径啦！
    

- Jupyter Lab的修改操作和Jupyter Notebook流程相似，但是在细节上有些不同
    

1. 同样我们还是首先需要查看配置文件，在终端输入
    
    jupyter lab --generate-config
    
2. 找到配置文件所在的文件夹，打开配置文件
    
3. 修改配置文件时，用`Ctrl+F`快捷键查找，输入关键词，找到`# c.ServerApp.notebook_dir`，去掉注释。改为`c.ServerApp.notebook_dir = 'D:\\Adatascience（这里填自己想改的文件路径）'`
    
4. 之后的步骤和Jupyter Notebook修改配置文件的第七至第十步相同
    

### 2.2 使用虚拟环境[](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/0.4%20Jupyter%E7%9B%B8%E5%85%B3%E6%93%8D%E4%BD%9C.html#id4 "永久链接至标题")

​ 需要注意的是，Anaconda安装的虚拟环境和Jupyter Notebook运行需要的Kernel并不互通。那么我们该如何解决这个问题，并且如果我们想要切换内核（Change Kernel），该如何操作呢？

1. 将在Anaconda中创建的虚拟环境添加`ipykernel`
    

# 如果还没创建环境，在创建时要加上ipykernel
conda create -n env_name python=3.8 ipykernel
# 如果已经创建环境，在环境中安装ipykernel
pip install ipykernel

1. 将虚拟环境写进Jupyter
    

python -m ipykernel install --user --name env_name --display-name "env_name"

1. 在`Kernel`中更换添加的虚拟环境即可
    

![image-20220812171025719](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E9%9B%B6%E7%AB%A0/figures/image-20220812171025719.png)