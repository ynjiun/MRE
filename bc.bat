python3 -m blobconverter --onnx-model model.onnx --shaves 6 --version 2021.4 --optimizer-params '--data_type=FP16' --compile-params '-ip U8 -op FP16'
