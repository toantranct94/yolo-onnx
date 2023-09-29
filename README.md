# yolo-onnx
## Install dependencies
```bash
pip install -r requirements.txt
```
## Run inference with python runtime
```bash
python test.py --source test_video/test.mov --conf 0.6 --device cpu
```
#### Lite models
```bash
python test.py --model_detection models/plate_lite.pt --model_recognition models/character_lite.pt
```
## Deloy with TorchServe
#### Install TorchServe dependencies
```bash
git clone https://github.com/pytorch/serve
cd serve
```
- With CPU:
```bash
python ./ts_scripts/install_dependencies.py
```
- With GPU and Cuda 11.8:
```bash
python ./ts_scripts/install_dependencies.py --cuda=cu118
```
- Install dependencies
```bash
pip install torchserve torch-model-archiver
```
#### Create model archive file
```bash
cd ..
mkdir serve_model_store

torch-model-archiver -f --model-name plate --version 1.0 --serialized-file models/plate_lite.pt --handler serve_handler/plate_handler.py  --export-path serve_model_store
torch-model-archiver -f --model-name character --version 1.0 --serialized-file models/character_lite.pt --handler serve_handler/character_handler.py  --export-path serve_model_store
```
#### Start TorchServe and register the model with config file
```bash
torchserve --start --model-store serve_model_store --models character=character.mar plate=plate.mar --ts-config cfg/config.properties
```
#### Check inference API
```bash
curl "http://localhost:8081/models"
```
<details>
  <summary>Result in</summary>

    {
      "models": [
        {
          "modelName": "character",
          "modelUrl": "character.mar"
        },
        {
          "modelName": "plate",
          "modelUrl": "plate.mar"
        }
      ]
    }
</details>

#### Run inference
```bash
python request.py --client-batching
```
<details>
  <summary>Result in</summary>
 
    With Batch Size 4, FPS at frame number 100 is 13.4
    [
      {
        "num_plate": 1,
        "boxes": [
          [
            1221.6578369140625,
            447.3987121582031,
            1380.3450927734375,
            502.4010314941406
          ]
        ],
        "score": [
          0.7349867224693298
        ],
        "classes": [
          "plate"
        ],
        "license_plate": [
          "30A61235"
        ]
      },
      {
        "num_plate": 1,
        "boxes": [
          [
            1212.3612060546875,
            448.799560546875,
            1372.7303466796875,
            504.0987854003906
          ]
        ],
        "score": [
          0.7360736727714539
        ],
        "classes": [
          "plate"
        ],
        "license_plate": [
          "30A61235"
        ]
      },
      {
        "num_plate": 1,
        "boxes": [
          [
            1200.732421875,
            450.4124450683594,
            1364.0225830078125,
            507.4985046386719
          ]
        ],
        "score": [
          0.7395196557044983
        ],
        "classes": [
          "plate"
        ],
        "license_plate": [
          "30A61235"
        ]
      },
      {
        "num_plate": 1,
        "boxes": [
          [
            1191.5098876953125,
            451.9449462890625,
            1357.8824462890625,
            509.03741455078125
          ]
        ],
        "score": [
          0.7451594471931458
        ],
        "classes": [
          "plate"
        ],
        "license_plate": [
          "30A61235"
        ]
      }
    ]
</details>

- To set batch_size = 10, we use the following command
```bash
python request.py --client-batching --batch_size 10
```

- If you have a camera connected, you can run inference on real time video from the camera as follows
```bash
python request.py --client-batching --batch_size --input 0
```

