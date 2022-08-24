To duplicate the following error:

	blob.version= Version.VERSION_2021_4
	Inputs
	Name: right, Type: DataType.U8F, Shape: [320, 180, 3, 1]
	Name: left, Type: DataType.U8F, Shape: [320, 180, 3, 1]
	Outputs
	Name: output, Type: DataType.FP16, Shape: [320, 180, 2, 1]
	[184430106183EB0F00] [266.144] [NeuralNetwork(8)] [critical] Fatal error in openvino '2021.4'. Likely because the model was compiled for different openvino version. If you want to select an explicit openvino version use: setOpenVINOVersion while creating pipeline. If error persists please report to developers. Log: 'softMaxNClasses' '157'
	[184430106183EB0F00] [269.508] [system] [critical] Fatal error. Please report to developers. Log: 'Fatal error on MSS CPU: trap: 00, address: 00000000' '0'
	
Run under depthAI virtual environment:
	python mre.py
	
The model.blob is compiled from model.onnx with following parameters:
	http://blobconverter.luxonis.com/
	OpenVINO version: 2021.4
	model source: ONNX Model
	Model file: model.onnx
	Model optimizer params: --data_type=FP16
	MyriadX compile params: -ip U8 -op FP16
	Shaves: 6



