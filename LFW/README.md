## LFW test

This repo contains codes for  the [LFW](http://vis-www.cs.umass.edu/lfw/) face verification test.
Now , it supports for [centerface](https://github.com/ydwen/caffe-face) [sphereface](https://github.com/wy1iu/sphereface) [AMSoftmax](https://github.com/happynear/AMSoftmax) [arcface or insightface](https://github.com/deepinsight/insightface)

And you can easily use it to test your own models.

## Usage

### 1. Download the pre-aligned to 112x112 LFW dataset

This data is transformed from the dataset provided by [arcface](https://github.com/deepinsight/insightface). I convert it from mxnet to numpy which nolonger requires mxnet.

[lfw-112x112@BaiduDrive](https://pan.baidu.com/s/1uCOedn21j9ZDcm-7yYuhYA)

### 2. Test your model
Add some code in `run_verify.py` to setup your model

```
def model_yours(do_mirror):
    model_dir = '/path/to/your/model/'
    model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'weights.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size

        
def model_factory(name, do_mirror):
    model_dict = {
        'centerface':model_centerface, 
        'sphereface':model_sphereface, 
        'AMSoftmax':model_AMSoftmax, 
        'arcface':model_arcface,
        'model_yours':model_yours,  # add your model
    }
    model_func = model_dict[name]
    return model_func(do_mirror)
```


Run your test with:

```shell
python run_verify.py yours 
```

By default verification will be done with `cosine` distance measure, to test with `L2` distance, run with

```shell
python run_verify.py yours L2
```
Add to save testing time , the horizontal flip feature is not used, if you want to use it ,run with

```shell
python run_verify.py yours L2 1
```

### 3. Test other models.
First follow the github link to each project and download the *`caffe`*  version of the model. Extract files to the corresponding directory  under `models`. 

This is the result I got with default test configuration (flip not used, cosine distance). 

| model      | image size | LFW-tested       | Offical |
| ---------- | ---------- | ---------------- | ------- |
| centerface | 96x112     | 0.97850+-0.00689 | 0.9928  |
| sphereface | 96x112     | 0.98550+-0.00500 | 0.9930  |
| AMSoftmax  | 96x112     | 0.98417+-0.00512 | 0.9908  |
| arcface    | 112x112    | 0.99567+-0.00343 | 0.9974  |


