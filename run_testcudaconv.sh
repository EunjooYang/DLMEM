#!/bin/bash

#python test_cudaconv.py --data-provider cifar --test-range 6 --train-range 1-5 --data-path /tmp/cifar-10-py-colmajor --inner-size 32 --save-path /tmp/cudaconv_cifar --gpu 0 --layer-def ~/cuda-convnet2/layers/layers-cifar10-11pct.cfg --layer-params ~/cuda-convnet2/layers/layer-params-cifar10-11pct.cfg --mini 256

python memlayout_cudaconv.py --data-provider dummy-lr-3072 --test-range 1 --train-range 1 --data-path /tmp/cifar-10-py-colmajor --inner-size 0 --save-path /tmp/cudaconv_rand --gpu 0 --layer-def ~/cuda-convnet2/layers/layers-cifar10-11pct.cfg --layer-params ~/cuda-convnet2/layers/layer-params-cifar10-11pct.cfg  --mini 64 --epochs 1
