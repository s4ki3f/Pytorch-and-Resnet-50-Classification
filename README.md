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
  
  now we will run a sample execution using exc.py
  
  
#transformations

```python
import torchvision
import numpy as np
nor1 = [0.5, 0.5, 0.5]
nor2 = [0.5, 0.5, 0.5]
resize_factor = 224

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize_factor),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(nor1, nor2)
])
```

#Creation of train and test dataset
```python
from torchvision.datasets import STL10
trainset, testset = STL10('/content',transform=transform, download = True), STL10('/content',"test",transform=transform)
```

#Dataloaders Initialization

```python
trainloader, testloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False), torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False)
```

#Dataloader testing

```python
classes = list()
import matplotlib
class_present = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

for i in class_present:
  classes.append(i)

def imshow(img):
     num = 0.5
     temp = 2
     one =1
     zero, two = 0,2
     img = img / temp + num
     npimg = img.numpy()
     matplotlib.pyplot.imshow(np.transpose(npimg, (one, temp, zero)))
     matplotlib.pyplot.show()

dataiter = iter(testloader)
images, labels = next(dataiter)   
imshow(torchvision.utils.make_grid(images))
one = 1
print(' '.join('%5s' % classes[labels[j]] for j in range(one)))
```

![download](https://user-images.githubusercontent.com/29111757/214787089-679e32f5-b423-47ac-b6d7-395eb5624909.png)


#Training Dataset Creation to feed that data into SVM Classifier

```python
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    newmodel.to('cuda')
```
#Here features are extracted from the training set and stored in x_train

```python
range_num = 1
for i in range(range_num):
  with torch.no_grad():
    for data ,label in trainloader:
      data = data.to(device)
      output = newmodel(data)
      np_arr, np_arrlabel = output.cpu().detach().numpy(), label.cpu().detach().numpy()
      x_train.append(np_arr)
      y_train.append(np_arrlabel)
```
and then, adding training set to numpy array
```python
x_train = np.array(x_train)
```

```python
X_trainv2 = list()

for i in range(5000):
  x = x_train[i]
  x = np.ravel(x)
  X_trainv2.append(x)
```
```python
X_trainv2, y_train = np.array(X_trainv2), np.array(y_train)
```
#Here features are extracted from the training set and stored in x_test

```python
x_test, y_test = list(), list()

for i in range(1):
  with torch.no_grad():
    for data ,label in testloader:
      data = data.to(device)
      output = newmodel(data)
      np_arr,np_arrlabel  = output.cpu().detach().numpy(), label.cpu().detach().numpy()
      x_test.append(np_arr)
      y_test.append(np_arrlabel)
```

adding test set to numpy array

```python
X_test = list()

for i in range(8000):
  x = x_test[i]
  x = np.ravel(x)
  X_test.append(x)

X_test, y_test = np.array(X_test), np.array(y_test)
```
#SVM
SVM is created and optimized for tuning efficiency and trained on features extracted from the trainset
Apply grid search cross validation for finding the best set of parameters through defining parameter range

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

l1 = [0.1, 1, 10, 100]
l2 = [1, 0.1, 0.01, 0.001]
l3 = ['poly']
# defining parameter range
param_grid = {'C': l1,
              'gamma': l2,
              'kernel': l3}

# apply grid search cross validation for finding the best set of parameters
three = 3
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = three)
grid.fit(X_trainv2, y_train)
```
Train SVM Model on best parameters. 
(SVM accuracy by applying bigram and Chi-square feature selection with 70% feature reduction increased 0.95% has an accuracy of 97.35%. SVM accuracy by applying unigram and Chi-square feature selection with 90% reduction features increased by 1.58% with the highest accuracy value 97.98%)

```python
from sklearn.svm import SVC
gamma_param = 0.01
C_param = 1
clf2 = SVC(kernel='poly', C=C_param, gamma=gamma_param, probability=True).fit(X_trainv2,y_train)
```
Code for finding the accuracy using code from Scratch

```python
def accuracy(y_test, pred):
  ans = 0
  num = 1
  for actual_value, predicted_value in zip(y_test, pred):
    if(actual_value == predicted_value):
      ans = ans+num
  value, value1 = 8000, 100 
  final = (ans/value)*value1 
  return final
```

Calculation of one vs one confusion matrix having parameters such as TP,FP,TN,FN 
```python
def confusionmatrixperclass(pc,nc,y_test, pred):
  fp, fn, tp, tn = 0, 0, 0, 0

  for actual_value, predicted_value in zip(y_test, pred):
      num = 1
      if predicted_value == pc and actual_value == pc:
        tp = tp+num
      elif predicted_value == pc and actual_value != pc:
        fp = fp+num
      elif predicted_value == nc and actual_value == nc:
        tn = tn+num
      elif predicted_value == nc and actual_value != nc:
        fn = fn+num
  cm = [
      [tn, fp],
      [fn, tp]
  ]
  return cm 
```

```python
from sklearn.metrics import confusion_matrix
```

importing pandas, seaborn, matplotlkib for ploting and visual representation

```python
import pandas as pd
import numpy as np
import seaborn 
import matplotlib

def calculate_tpr_fpr(y_real, y_pred):

    cm = confusion_matrix(y_real, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    tpr, fpr =  TP/(TP + FN), 1 - TN/(TN+FP) 
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    uyt = 0
    tpr_list, fpr_list = [uyt], [uyt]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tre1, tre2 = y_real, y_pred
        tpr, fpr = calculate_tpr_fpr(tre1, tre2)
        tpr_final, fpr_final = tpr, fpr
        tpr_list.append(tpr_final)
        fpr_list.append(fpr_final)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    if ax == None:
        five = 5
        matplotlib.pyplot.figure(figsize = (five, five))
        ax = matplotlib.pyplot.axes()
    if scatter:
        seaborn.scatterplot(x = fpr, y = tpr, ax = ax)
    seaborn.lineplot(x = fpr, y = tpr, ax = ax)
    xx, yy = -0.05, 1.05
    xer, one = 0,1
    seaborn.lineplot(x = [xer, one], y = [xer, one], color = 'green', ax = ax)
    matplotlib.pyplot.xlim(xx,yy)
    matplotlib.pyplot.ylim(xx,yy)
    matplotlib.pyplot.xlabel("False Positive Rate")
    matplotlib.pyplot.ylabel("True Positive Rate")
```

```python
prob, pred = clf2.predict_proba(X_test), clf2.predict(X_test)
classes = clf2.classes_
pred, prob = np.array(pred), np.array(prob)
def func(c,y_test):
 list_comp = [1 if i == c else 0 for i in y_test]
 return list_comp
```

ploting
```python
matplotlib.pyplot.figure(figsize = (12, 8))
from sklearn.metrics import roc_auc_score
bins = [i/20 for i in range(20)] + [1]
roc_auc_ovr = dict()

for i in range(len(classes)):
    c = classes[i]
    val1, val2 = func(c,y_test), prob[:, i]    
    two, five, one = 2,5,1
    ax_bottom = matplotlib.pyplot.subplot(two, five, i+one)
    tpr, fpr = get_all_roc_coordinates(val1, val2)
    val = False
    plot_roc_curve(tpr, fpr, scatter = val, ax = ax_bottom)
    ax_bottom.set_title("ROC Curve")
    
    roc_auc_ovr[c] = roc_auc_score(val1, val2)
    
matplotlib.pyplot.tight_layout()
```
![download (1)](https://user-images.githubusercontent.com/29111757/214788587-096ffbcc-33a8-4727-8633-49a74464c9a1.png)

#Class accuracy confusion matrix's output

```python
ten_num = 10
for i in range(0,ten_num):
  for j in range(0,ten_num):
    cm = confusionmatrixperclass(i,j,y_test,pred)
    print("confusion matrix for positive class ",i," and negative class ",j,"\n")
    print(cm)
    print("\n")
```

