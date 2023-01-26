# s4ki3f-Pytorch-and-Resnet-50-Classification

<br>
<div>
  <a href="https://colab.research.google.com/drive/13tE4SB8b4T-v25EeyjNDHIh0NPJPb6GP#scrollTo=8a-S_n5uG8F0"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>

## Importing pytorch and downloading the model

```python
import torch
var1 = True
neg_num = -1
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=var1)
newmodel = torch.nn.Sequential(*(list(model.children())[:neg_num]))
```

*Import* all the necessary libraries

```python
import urllib as path_files
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: 
  path_files.URLopener().retrieve(url, filename)
except: 
  path_files.request.urlretrieve(url, filename)
  ```
  
  now we will run a sample execution
