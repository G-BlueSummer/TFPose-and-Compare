# post-processing for Part-Affinity Fields Map implemented in C++ & Swig

需要先安装C++编译器与swig，然后执行以下代码进行编译

```bash
$ swig -python -c++ pafprocess.i && python setup.py build_ext --inplace
```

