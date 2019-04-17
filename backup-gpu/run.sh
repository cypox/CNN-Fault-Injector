#!/bin/bash

export LD_LIBRARY_PATH=/opt/caffe/ristretto/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/caffe/ristretto/python/:$PYTHONPATH

export PARENT=/home/aiembed/date-iccd-paper
export NETWORKS=/home/aiembed/date-iccd-paper
export DATASET=/media/aiembed/cf11bc1b-0f02-4383-ad36-da07690b0deb/ilsvrc/validation

rm -f outputs/output-$1
make

export MAXITER=0

#./fault-injector $NETWORKS/$1/$1_quantized_test.prototxt $NETWORKS/$1/$1_finetuned.caffemodel $NETWORKS/$1/$1'_parsed' $DATASET $2 1000 > outputs/output-$1

for i in `seq 0 4`;
do
  let "a = 10**$i";
  echo "Injecting $a faults";
  ./fault-injector $NETWORKS/$1/$1_quantized_test.prototxt $NETWORKS/$1/$1_finetuned.caffemodel $NETWORKS/$1/$1'_parsed' $DATASET $a $MAXITER > outputs/n$a-output-$1;
done

