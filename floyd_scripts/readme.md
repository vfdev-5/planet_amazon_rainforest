

## How to launch training on floydhub

```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee "python train.py"
```

```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee "python val_predict.py"
```


## How to launch predictions on floydhub

```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee:input_1 --data NkXWSWfHP83nemMueZjiK4:input_2 "python predict.py"
```

## Mode jupyter on trained model

```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee:input --data CDnxrZEsrxuHTWU5tKqWve:ws --mode jupyter
```

```
!mkdir -p /output/ws/
!cp -R /ws/planet_amazon_rainforest   /output/ws/
!cd /output/ws/planet_amazon_rainforest && git pull origin master

!ln -s /ws/generated /output/ws/generated
 !ln -s /ws/weights /output/ws/weights

os.environ['INPUT_PATH'] = '/input'
os.environ['OUTPUT_PATH'] = '/output/ws'
```