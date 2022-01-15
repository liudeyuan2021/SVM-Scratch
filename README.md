### SVM Scratch

训练代码使用Python，测试代码将使用C++重写

#### 1 参考代码
参考sklearn的Python和C++的代码实现了SVM测试部分的python代码，基本无任何包依赖，可较方便地改写为C++代码集成到项目中，主要参考代码如下：
+ [sklearn/svm/_libsvm.pxi](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_libsvm.pxi#L9)
+ [sklearn/svm/_libsvm.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_libsvm.pyx#L283)
+ [sklearn/svm/src/libsvm/libsvm_helper.c](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/src/libsvm/libsvm_helper.c#L114)
+ [sklearn/svm/src/libsvm/svm.h](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/src/libsvm/svm.h#L46)
+ [sklearn/svm/src/libsvm/svm.cpp](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/src/libsvm/svm.cpp#L2818)
+ [sklearn/utils/_cython_blas.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_cython_blas.pyx#L20)