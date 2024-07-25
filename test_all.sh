#!/bin/sh
nohup python -u test_pf.py --cfg a03 --gpu 0 > ./outs/test_a03.out 2>&1 & 
nohup python -u test_pf.py --cfg a04 --gpu 0 > ./outs/test_a04.out 2>&1 & 
nohup python -u test_pf.py --cfg a07 --gpu 0 > ./outs/test_a07.out 2>&1 & 
nohup python -u test_pf.py --cfg a08 --gpu 0 > ./outs/test_a08.out 2>&1 & 

nohup python -u test_pf.py --cfg d03 --gpu 1 > ./outs/test_d03.out 2>&1 & 
nohup python -u test_pf.py --cfg d04 --gpu 1 > ./outs/test_d04.out 2>&1 & 
nohup python -u test_pf.py --cfg d07 --gpu 1 > ./outs/test_d07.out 2>&1 & 
nohup python -u test_pf.py --cfg d08 --gpu 1 > ./outs/test_d08.out 2>&1 & 

nohup python -u test_pf.py --cfg m03 --gpu 0 > ./outs/test_m03.out 2>&1 & 
nohup python -u test_pf.py --cfg m04 --gpu 1 > ./outs/test_m04.out 2>&1 & 
nohup python -u test_pf.py --cfg m07 --gpu 0 > ./outs/test_m07.out 2>&1 & 
nohup python -u test_pf.py --cfg m08 --gpu 1 > ./outs/test_m08.out 2>&1 & 