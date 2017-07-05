(* ::Package:: *)

(* Execute only from the directory containing training
inputs and masks *)

net = Import["./checkpoints/trained.wlnet"]

meanImage = Import["meanImage.jpg"]

validFile = Import["valData.mx"]

meanImage = ImageData[ImageResize[meanImage, {256, 256}], Interleaving -> False];

imgFiles = File /@ FileNames["train/pics/*.png"];
maskPaths = FileNames["train/labels/*.png"];

importerFast = NeuralNetworks`Private`ToEncoderFunction[
   NetEncoder[{"Image", {256, 256}}], True];

importerSlow =
  ImageData[First@Image`ImportExportDump`ImageReadPNG[#], "Byte"] +
    1 &;

batchSize = 16;

nData = Length@imgFiles;

meanImageList = ConstantArray[meanImage, batchSize]

generator = Function[
   Block[
   		{inputs, outputs, minibatch},

		If[i + #BatchSize - 1 > nData, i = 1];
	    inputs = importerFast@imgFiles[[i ;; i + #BatchSize - 1]];

		inputs = inputs - meanImageList;

	    outputs = importerSlow /@ maskPaths[[i ;; i + #BatchSize - 1]];
		minibatch = Thread[inputs -> outputs];
	    i += #BatchSize;
	    minibatch
    ]
  ];

  getValidData := Function[
  	{set},

  	Block[
  		{maskData},
  		maskData = ImageData[set[[2]], "Byte"];
  		set[[1]] -> maskData
  	]
  ]

validData = getValidData /@ validFile

i = 1;

trainedNet = NetTrain[net, generator, BatchSize -> batchSize, MaxTrainingRounds -> 1000000000, TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"Directory", "./checkpoints", "Interval" -> Quantity[30, "Minutes"]}]

Export["./trainedNet.wlnet", trainedNet]
