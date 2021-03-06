# 1. 简介
本仓库为CRF++工具的python接口封装，提供以下功能：
+ 序列标注标签体系生成
+ 利用命令行执行脚本训练模型
+ 使用交叉验证法选取最优超参
+ Python接口封装进行序列标注预测


## 1.1 CRF++: Yet Another CRF toolkit
+ 官方主页: https://taku910.github.io/crfpp/#download
+ github网址: https://github.com/taku910/crfpp
### 安装CRF++
从官方主页中下载最新CRF++-0.58版本，并解压。进入到./CRF++-0.58/目录下，安装步骤如下：
```
./configure 
make
su
make install
```
### 安装python库：CRFPP
进入到./CRF++-0.58/python/目录下，执行命令：
```
python3 setup.py build
python3 setup.py install
```
## 1.2 Python Wrapper for CRF++
在./CRF++-0.58/src/目录下，包含模块文件：
+ crf_wrapper.py: python wrapper for CRF++，加载训练好的模型进行序列标注预测；
+ utils.py: 提供一些公共函数，如生成序列标注标签、生成训练输入文件等；
+ evaluate.py: 进行分类指标评价(如f1)；

# 2. 示例
CRF++工具可以用于各类序列标注任务，如分词、词性标注、命名实体识别(NER)等。在./pycrfpp/examples/目录下，每个序列标注项目(如ner/)文件夹下应至少包含以下文件：
+ run.sh 训练、测试、评估的脚本文件
+ template 特征模板文件
+ train.data 训练集输入文件
+ test.data 测试集输入文件

## 2.1 详细步骤
+ 提取文本特征，准备训练数据文件
  - ./train.data
  - ./test.data
+ 准备特征模板文件
  - ./template
+ 训练模型：在训练集上训练，输出模型文件crfpp_model.bin
```
crf_learn -p 4 -c 1 -f 1 template train.data crfpp_model.bin
```
+ 测试结果：在测试集上测试，输出test.result
```
crf_test -m crfpp_model.bin test.data > test.result
```
+ 最后，利用CrfWrapper类(./pycrfpp/crf_wrapper.py)加载训练后的模型进行序列标注预测