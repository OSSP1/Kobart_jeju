# Korean Dialect <-> Jeju Standard Translator

## How to train

* hyperparameter
  
```
* epochs = 10, gpus = 1, learning_rate = 2e-7, batch_size = 32
```
### dialect > standard

````
python trainer_d2s.py
````
* 훈련을 마친 후 모델이 model_results/d2s/model 에 저장됩니다.


### standard > dialect

````
python trainer_s2d.py
````
* 훈련을 마친 후 모델이 model_results/2ds/{region}/model 에 저장됩니다.


## How to generate

* inference_example.ipynb
  * 생성과 스코어링 예시를 담고 있습니다.


## add.. 

cuda version 12.1

````
conda create -n 환경명 anaconda
````

torch 설치 
````
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
````
torch 1.13.1+cu117

pytorch-lightning 설치
````
pip install pytorch-ligntning==1.2.1
````







