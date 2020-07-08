# 简介
本仓库为CRF++工具的python接口封装，主要提供以下功能：
+ 序列标注标签体系生成
+ 利用命令行执行脚本，训练模型
+ 使用交叉验证法选取最优超参
+ Python接口封装，进行序列标注预测


## CRF++: Yet Another CRF toolkit
+ 官方主页: https://taku910.github.io/crfpp/#download
+ github网址: https://github.com/taku910/crfpp
### 安装CRF++
从官方主页中下载CRF++-0.58版本，解压。进入到./CRF++-0.58，安装过程如下：
```
% ./configure 
% make
% su
# make install
```
### 安装python包：CRFPP
进入到./CRF++-0.58/python/目录
```
python3 setup.py build
python3 setup.py install
```
## Python Wrapper

# 示例
每个项目(project)文件夹下应至少包含以下文件：
+ run.sh 运行脚本文件
+ template 模板文件
+ train.data 训练集输入文件
+ test.data 测试集输入文件

## 详细步骤
+ 1.提取文本特征，准备训练数据
```
train.data
test.data
```
+ 2.模板文件template
```
template
```
+ 3.训练模型:
```
crf_learn -p 4 -c 10 -f 3 template train.data crfpp_model.bin
```
+ 4.测试：输出test.result
```
crf_test -m crfpp_model.bin test.data > test.result
```

