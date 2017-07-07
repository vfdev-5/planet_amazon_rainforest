

## How to launch training on floydhub

```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee "python start_all.py"
```
or EOL dependent falsy version
```
floyd run --env tensorflow-1.2 --gpu --data Kr5VngRdzGWSUxD7dCtkee "sh start_all.sh"
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