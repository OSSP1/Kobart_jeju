# Korean Dialect <-> Standard Translator

## How to train

* hyperparameter는 epochs = 3, gpus = 1 로 지정되어있습니다.

### dialect > standard

````
python trainer_d2s.py
````
* 훈련을 마친 후 모델이 model_results/d2s/{region}/model 에 저장됩니다.


### standard > dialect

````
python3 trainer_s2d.py --region 'jeju'
````
* 훈련을 마친 후 모델이 model_results/2ds/{region}/model 에 저장됩니다.


## How to generate

* inference_example.ipynb 를 참조해주세요 :)
  * 생성과 스코어링 예시를 담고 있습니다.


## add.. 

cuda version 12.1

가상환경 새로 만들기
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







