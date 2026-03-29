#!/bin/bash
mkdir -p log
for n in $(seq 1 100); do
    make run > "log/${n}.txt"
done