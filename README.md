# MobileNetV3-MXNET-insightface

How to use
==========
pip insatll mxnet-cu90     (updated to version 1.5.0)

cp fmobilenetv3.py ../insightface/src/symbols/

cp m3_train_softmax.py ../insight/src/

train:
cd ../insightface/src/
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u m3_train_softmax.py --data-dir your_data_dir --network l3 --loss-type 0 --prefix your_savemodel_dir --per-batch-size 128

Solve the problem of model oversize
---------------
cd ../insightface/deploy/
python -u model-slim.py --model your_savemodel_dir/,epoch
