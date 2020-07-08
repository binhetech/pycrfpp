crf_learn -p 4 -c 10 -f 3 template-323 train.data crfpp_model.bin
crf_test -m crfpp_model.bin test.data > test.result
python3 ../../src/evaluate.py -file test.result -head 2 | tee test.log
