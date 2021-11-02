```python
%matplotlib inline
```


```python
import torch
import torchvision
import torchvision.transforms as transforms
```


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
```

    cuda:0
    


```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomHorizontalFlip(p=1),
     transforms.RandomVerticalFlip(p=1),
     transforms.RandomRotation(degrees = 45)
     ])

# transform_test=[transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
    


    HBox(children=(FloatProgress(value=0.0, max=170498071.0), HTML(value='')))


    
    Extracting ./data/cifar-10-python.tar.gz to ./data
    Files already downloaded and verified
    


```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


```


![png](./best_attempt%28model7%29_4_0.png)


     frog plane  ship horse
    


```python
for i, data in enumerate(trainloader, 0):
  inputs, labels = data[0].to(device), data[1].to(device)

inputs.shape
```




    torch.Size([4, 3, 32, 32])




```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*5*5, 1600)
        self.fc2 = nn.Linear(1600, 800)
        self.fc3 = nn.Linear(800,400)
        self.fc4 = nn.Linear(400,200)
        self.fc5 = nn.Linear(200,100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


net = Net()
net=net.to(device)
print(net)
```

    Net(
      (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=3200, out_features=1600, bias=True)
      (fc2): Linear(in_features=1600, out_features=800, bias=True)
      (fc3): Linear(in_features=800, out_features=400, bias=True)
      (fc4): Linear(in_features=400, out_features=200, bias=True)
      (fc5): Linear(in_features=200, out_features=100, bias=True)
      (fc6): Linear(in_features=100, out_features=50, bias=True)
      (fc7): Linear(in_features=50, out_features=10, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    


```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


```python
# PATH='./cifar_net.pth'
# nettorch.load(PATH)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    total_train = 0
    correct_train = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # accuracy source: https://discuss.pytorch.org/t/calculate-train-accuracy-of-the-model-in-segmentation-task/33581
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        #avg_accuracy = train_accuracy / len(train_loader)


        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            print("training accuracy : ",train_accuracy)
            running_loss = 0.0

print('Finished Training')
```

    [1,  2000] loss: 2.304
    training accuracy :  10.35
    [1,  4000] loss: 2.303
    training accuracy :  10.075
    [1,  6000] loss: 2.302
    training accuracy :  10.233333333333333
    [1,  8000] loss: 2.278
    training accuracy :  11.08125
    [1, 10000] loss: 2.068
    training accuracy :  12.5025
    [1, 12000] loss: 1.978
    training accuracy :  14.08125
    [2,  2000] loss: 1.888
    training accuracy :  24.6375
    [2,  4000] loss: 1.834
    training accuracy :  25.99375
    [2,  6000] loss: 1.785
    training accuracy :  27.3125
    [2,  8000] loss: 1.768
    training accuracy :  28.525
    [2, 10000] loss: 1.706
    training accuracy :  29.6025
    [2, 12000] loss: 1.653
    training accuracy :  30.6375
    [3,  2000] loss: 1.605
    training accuracy :  39.2875
    [3,  4000] loss: 1.565
    training accuracy :  40.2625
    [3,  6000] loss: 1.544
    training accuracy :  41.02916666666667
    [3,  8000] loss: 1.515
    training accuracy :  41.6625
    [3, 10000] loss: 1.476
    training accuracy :  42.445
    [3, 12000] loss: 1.455
    training accuracy :  43.15416666666667
    [4,  2000] loss: 1.397
    training accuracy :  48.1
    [4,  4000] loss: 1.364
    training accuracy :  49.50625
    [4,  6000] loss: 1.335
    training accuracy :  50.233333333333334
    [4,  8000] loss: 1.356
    training accuracy :  50.584375
    [4, 10000] loss: 1.309
    training accuracy :  50.9775
    [4, 12000] loss: 1.312
    training accuracy :  51.44166666666667
    [5,  2000] loss: 1.241
    training accuracy :  55.875
    [5,  4000] loss: 1.236
    training accuracy :  55.95625
    [5,  6000] loss: 1.222
    training accuracy :  56.083333333333336
    [5,  8000] loss: 1.199
    training accuracy :  56.496875
    [5, 10000] loss: 1.180
    training accuracy :  57.0025
    [5, 12000] loss: 1.170
    training accuracy :  57.37708333333333
    [6,  2000] loss: 1.132
    training accuracy :  60.775
    [6,  4000] loss: 1.113
    training accuracy :  60.825
    [6,  6000] loss: 1.095
    training accuracy :  61.0625
    [6,  8000] loss: 1.112
    training accuracy :  61.21875
    [6, 10000] loss: 1.081
    training accuracy :  61.515
    [6, 12000] loss: 1.069
    training accuracy :  61.75833333333333
    [7,  2000] loss: 1.007
    training accuracy :  65.2625
    [7,  4000] loss: 1.011
    training accuracy :  65.025
    [7,  6000] loss: 1.019
    training accuracy :  65.04166666666667
    [7,  8000] loss: 1.009
    training accuracy :  65.19375
    [7, 10000] loss: 0.996
    training accuracy :  65.2075
    [7, 12000] loss: 1.029
    training accuracy :  65.14583333333333
    [8,  2000] loss: 0.946
    training accuracy :  67.375
    [8,  4000] loss: 0.897
    training accuracy :  68.2375
    [8,  6000] loss: 0.942
    training accuracy :  68.16666666666667
    [8,  8000] loss: 0.952
    training accuracy :  68.025
    [8, 10000] loss: 0.936
    training accuracy :  68.07
    [8, 12000] loss: 0.951
    training accuracy :  68.09166666666667
    [9,  2000] loss: 0.877
    training accuracy :  70.3875
    [9,  4000] loss: 0.871
    training accuracy :  70.58125
    [9,  6000] loss: 0.844
    training accuracy :  70.9375
    [9,  8000] loss: 0.878
    training accuracy :  70.58125
    [9, 10000] loss: 0.891
    training accuracy :  70.3825
    [9, 12000] loss: 0.877
    training accuracy :  70.40208333333334
    [10,  2000] loss: 0.796
    training accuracy :  72.6
    [10,  4000] loss: 0.803
    training accuracy :  72.9375
    [10,  6000] loss: 0.813
    training accuracy :  72.80416666666666
    [10,  8000] loss: 0.815
    training accuracy :  72.8
    [10, 10000] loss: 0.819
    training accuracy :  72.7275
    [10, 12000] loss: 0.824
    training accuracy :  72.62708333333333
    [11,  2000] loss: 0.745
    training accuracy :  74.7875
    [11,  4000] loss: 0.755
    training accuracy :  74.49375
    [11,  6000] loss: 0.752
    training accuracy :  74.50416666666666
    [11,  8000] loss: 0.763
    training accuracy :  74.371875
    [11, 10000] loss: 0.775
    training accuracy :  74.2475
    [11, 12000] loss: 0.783
    training accuracy :  74.07916666666667
    [12,  2000] loss: 0.682
    training accuracy :  76.875
    [12,  4000] loss: 0.717
    training accuracy :  76.36875
    [12,  6000] loss: 0.715
    training accuracy :  76.07083333333334
    [12,  8000] loss: 0.719
    training accuracy :  76.04375
    [12, 10000] loss: 0.710
    training accuracy :  75.965
    [12, 12000] loss: 0.740
    training accuracy :  75.84583333333333
    [13,  2000] loss: 0.625
    training accuracy :  79.3875
    [13,  4000] loss: 0.637
    training accuracy :  78.9625
    [13,  6000] loss: 0.660
    training accuracy :  78.5375
    [13,  8000] loss: 0.675
    training accuracy :  78.059375
    [13, 10000] loss: 0.658
    training accuracy :  77.9725
    [13, 12000] loss: 0.691
    training accuracy :  77.74375
    [14,  2000] loss: 0.614
    training accuracy :  79.1
    [14,  4000] loss: 0.618
    training accuracy :  79.00625
    [14,  6000] loss: 0.616
    training accuracy :  79.1125
    [14,  8000] loss: 0.637
    training accuracy :  78.984375
    [14, 10000] loss: 0.632
    training accuracy :  78.9125
    [14, 12000] loss: 0.640
    training accuracy :  78.86458333333333
    [15,  2000] loss: 0.556
    training accuracy :  81.0125
    [15,  4000] loss: 0.561
    training accuracy :  81.075
    [15,  6000] loss: 0.568
    training accuracy :  81.15416666666667
    [15,  8000] loss: 0.590
    training accuracy :  80.95
    [15, 10000] loss: 0.595
    training accuracy :  80.705
    [15, 12000] loss: 0.593
    training accuracy :  80.55208333333333
    [16,  2000] loss: 0.521
    training accuracy :  82.7125
    [16,  4000] loss: 0.540
    training accuracy :  82.3
    [16,  6000] loss: 0.536
    training accuracy :  82.03333333333333
    [16,  8000] loss: 0.534
    training accuracy :  82.08125
    [16, 10000] loss: 0.566
    training accuracy :  81.915
    [16, 12000] loss: 0.547
    training accuracy :  81.90833333333333
    [17,  2000] loss: 0.482
    training accuracy :  83.7
    [17,  4000] loss: 0.499
    training accuracy :  83.675
    [17,  6000] loss: 0.513
    training accuracy :  83.375
    [17,  8000] loss: 0.510
    training accuracy :  83.24375
    [17, 10000] loss: 0.511
    training accuracy :  83.1475
    [17, 12000] loss: 0.519
    training accuracy :  83.03333333333333
    [18,  2000] loss: 0.429
    training accuracy :  85.45
    [18,  4000] loss: 0.469
    training accuracy :  84.85
    [18,  6000] loss: 0.466
    training accuracy :  84.80416666666666
    [18,  8000] loss: 0.480
    training accuracy :  84.64375
    [18, 10000] loss: 0.496
    training accuracy :  84.43
    [18, 12000] loss: 0.519
    training accuracy :  84.15
    [19,  2000] loss: 0.445
    training accuracy :  85.1125
    [19,  4000] loss: 0.415
    training accuracy :  85.75
    [19,  6000] loss: 0.422
    training accuracy :  85.64583333333333
    [19,  8000] loss: 0.455
    training accuracy :  85.446875
    [19, 10000] loss: 0.456
    training accuracy :  85.3675
    [19, 12000] loss: 0.465
    training accuracy :  85.26875
    [20,  2000] loss: 0.405
    training accuracy :  86.425
    [20,  4000] loss: 0.387
    training accuracy :  86.89375
    [20,  6000] loss: 0.408
    training accuracy :  86.6
    [20,  8000] loss: 0.433
    training accuracy :  86.534375
    [20, 10000] loss: 0.405
    training accuracy :  86.53
    [20, 12000] loss: 0.435
    training accuracy :  86.38333333333334
    [21,  2000] loss: 0.355
    training accuracy :  88.725
    [21,  4000] loss: 0.383
    training accuracy :  88.13125
    [21,  6000] loss: 0.375
    training accuracy :  87.88333333333334
    [21,  8000] loss: 0.407
    training accuracy :  87.55625
    [21, 10000] loss: 0.397
    training accuracy :  87.38
    [21, 12000] loss: 0.401
    training accuracy :  87.25208333333333
    [22,  2000] loss: 0.357
    training accuracy :  88.225
    [22,  4000] loss: 0.347
    training accuracy :  88.21875
    [22,  6000] loss: 0.369
    training accuracy :  88.08333333333333
    [22,  8000] loss: 0.386
    training accuracy :  87.853125
    [22, 10000] loss: 0.377
    training accuracy :  87.7775
    [22, 12000] loss: 0.381
    training accuracy :  87.63541666666667
    [23,  2000] loss: 0.322
    training accuracy :  89.125
    [23,  4000] loss: 0.340
    training accuracy :  88.70625
    [23,  6000] loss: 0.333
    training accuracy :  88.82083333333334
    [23,  8000] loss: 0.334
    training accuracy :  88.934375
    [23, 10000] loss: 0.365
    training accuracy :  88.76
    [23, 12000] loss: 0.359
    training accuracy :  88.64375
    [24,  2000] loss: 0.310
    training accuracy :  90.0875
    [24,  4000] loss: 0.304
    training accuracy :  90.0125
    [24,  6000] loss: 0.331
    training accuracy :  89.675
    [24,  8000] loss: 0.325
    training accuracy :  89.528125
    [24, 10000] loss: 0.346
    training accuracy :  89.375
    [24, 12000] loss: 0.330
    training accuracy :  89.37083333333334
    [25,  2000] loss: 0.281
    training accuracy :  91.025
    [25,  4000] loss: 0.312
    training accuracy :  90.3375
    [25,  6000] loss: 0.279
    training accuracy :  90.52083333333333
    [25,  8000] loss: 0.286
    training accuracy :  90.55
    [25, 10000] loss: 0.294
    training accuracy :  90.4575
    [25, 12000] loss: 0.325
    training accuracy :  90.29375
    [26,  2000] loss: 0.272
    training accuracy :  90.8125
    [26,  4000] loss: 0.272
    training accuracy :  90.9625
    [26,  6000] loss: 0.287
    training accuracy :  90.8625
    [26,  8000] loss: 0.291
    training accuracy :  90.83125
    [26, 10000] loss: 0.291
    training accuracy :  90.7525
    [26, 12000] loss: 0.304
    training accuracy :  90.55208333333333
    [27,  2000] loss: 0.257
    training accuracy :  91.5125
    [27,  4000] loss: 0.237
    training accuracy :  91.80625
    [27,  6000] loss: 0.266
    training accuracy :  91.62083333333334
    [27,  8000] loss: 0.268
    training accuracy :  91.446875
    [27, 10000] loss: 0.274
    training accuracy :  91.3175
    [27, 12000] loss: 0.282
    training accuracy :  91.30416666666666
    [28,  2000] loss: 0.241
    training accuracy :  92.2625
    [28,  4000] loss: 0.232
    training accuracy :  92.19375
    [28,  6000] loss: 0.237
    training accuracy :  92.32083333333334
    [28,  8000] loss: 0.265
    training accuracy :  92.125
    [28, 10000] loss: 0.260
    training accuracy :  91.9975
    [28, 12000] loss: 0.268
    training accuracy :  91.86458333333333
    [29,  2000] loss: 0.223
    training accuracy :  92.7875
    [29,  4000] loss: 0.240
    training accuracy :  92.625
    [29,  6000] loss: 0.235
    training accuracy :  92.59583333333333
    [29,  8000] loss: 0.234
    training accuracy :  92.546875
    [29, 10000] loss: 0.226
    training accuracy :  92.5325
    [29, 12000] loss: 0.252
    training accuracy :  92.41458333333334
    [30,  2000] loss: 0.210
    training accuracy :  92.9
    [30,  4000] loss: 0.231
    training accuracy :  92.6125
    [30,  6000] loss: 0.216
    training accuracy :  92.80416666666666
    [30,  8000] loss: 0.234
    training accuracy :  92.690625
    [30, 10000] loss: 0.222
    training accuracy :  92.715
    [30, 12000] loss: 0.229
    training accuracy :  92.64791666666666
    [31,  2000] loss: 0.209
    training accuracy :  93.2
    [31,  4000] loss: 0.213
    training accuracy :  93.1875
    [31,  6000] loss: 0.218
    training accuracy :  93.22083333333333
    [31,  8000] loss: 0.221
    training accuracy :  93.125
    [31, 10000] loss: 0.223
    training accuracy :  93.0375
    [31, 12000] loss: 0.217
    training accuracy :  92.99375
    [32,  2000] loss: 0.193
    training accuracy :  94.0375
    [32,  4000] loss: 0.195
    training accuracy :  93.84375
    [32,  6000] loss: 0.215
    training accuracy :  93.55833333333334
    [32,  8000] loss: 0.201
    training accuracy :  93.559375
    [32, 10000] loss: 0.197
    training accuracy :  93.5525
    [32, 12000] loss: 0.221
    training accuracy :  93.40833333333333
    [33,  2000] loss: 0.197
    training accuracy :  93.775
    [33,  4000] loss: 0.190
    training accuracy :  93.65
    [33,  6000] loss: 0.201
    training accuracy :  93.59166666666667
    [33,  8000] loss: 0.187
    training accuracy :  93.63125
    [33, 10000] loss: 0.208
    training accuracy :  93.5375
    [33, 12000] loss: 0.219
    training accuracy :  93.45
    [34,  2000] loss: 0.176
    training accuracy :  94.4125
    [34,  4000] loss: 0.177
    training accuracy :  94.525
    [34,  6000] loss: 0.175
    training accuracy :  94.44583333333334
    [34,  8000] loss: 0.187
    training accuracy :  94.2625
    [34, 10000] loss: 0.185
    training accuracy :  94.27
    [34, 12000] loss: 0.196
    training accuracy :  94.12083333333334
    [35,  2000] loss: 0.165
    training accuracy :  94.7875
    [35,  4000] loss: 0.172
    training accuracy :  94.66875
    [35,  6000] loss: 0.170
    training accuracy :  94.6875
    [35,  8000] loss: 0.176
    training accuracy :  94.575
    [35, 10000] loss: 0.171
    training accuracy :  94.4875
    [35, 12000] loss: 0.174
    training accuracy :  94.43541666666667
    [36,  2000] loss: 0.151
    training accuracy :  95.275
    [36,  4000] loss: 0.141
    training accuracy :  95.3875
    [36,  6000] loss: 0.161
    training accuracy :  95.25416666666666
    [36,  8000] loss: 0.173
    training accuracy :  95.021875
    [36, 10000] loss: 0.161
    training accuracy :  94.97
    [36, 12000] loss: 0.167
    training accuracy :  94.86458333333333
    [37,  2000] loss: 0.151
    training accuracy :  95.1125
    [37,  4000] loss: 0.161
    training accuracy :  95.0
    [37,  6000] loss: 0.165
    training accuracy :  94.84583333333333
    [37,  8000] loss: 0.165
    training accuracy :  94.79375
    [37, 10000] loss: 0.158
    training accuracy :  94.8225
    [37, 12000] loss: 0.160
    training accuracy :  94.80625
    [38,  2000] loss: 0.140
    training accuracy :  95.375
    [38,  4000] loss: 0.147
    training accuracy :  95.3
    [38,  6000] loss: 0.146
    training accuracy :  95.31666666666666
    [38,  8000] loss: 0.154
    training accuracy :  95.25
    [38, 10000] loss: 0.146
    training accuracy :  95.1975
    [38, 12000] loss: 0.160
    training accuracy :  95.125
    [39,  2000] loss: 0.150
    training accuracy :  95.2
    [39,  4000] loss: 0.139
    training accuracy :  95.4375
    [39,  6000] loss: 0.148
    training accuracy :  95.4
    [39,  8000] loss: 0.132
    training accuracy :  95.475
    [39, 10000] loss: 0.149
    training accuracy :  95.41
    [39, 12000] loss: 0.153
    training accuracy :  95.33541666666666
    [40,  2000] loss: 0.139
    training accuracy :  95.425
    [40,  4000] loss: 0.132
    training accuracy :  95.59375
    [40,  6000] loss: 0.144
    training accuracy :  95.575
    [40,  8000] loss: 0.141
    training accuracy :  95.540625
    [40, 10000] loss: 0.140
    training accuracy :  95.4725
    [40, 12000] loss: 0.139
    training accuracy :  95.4875
    [41,  2000] loss: 0.116
    training accuracy :  96.25
    [41,  4000] loss: 0.126
    training accuracy :  96.11875
    [41,  6000] loss: 0.133
    training accuracy :  95.975
    [41,  8000] loss: 0.131
    training accuracy :  95.946875
    [41, 10000] loss: 0.144
    training accuracy :  95.8625
    [41, 12000] loss: 0.129
    training accuracy :  95.90625
    [42,  2000] loss: 0.113
    training accuracy :  96.575
    [42,  4000] loss: 0.130
    training accuracy :  96.28125
    [42,  6000] loss: 0.118
    training accuracy :  96.22083333333333
    [42,  8000] loss: 0.130
    training accuracy :  96.075
    [42, 10000] loss: 0.125
    training accuracy :  96.065
    [42, 12000] loss: 0.129
    training accuracy :  96.05625
    [43,  2000] loss: 0.106
    training accuracy :  96.725
    [43,  4000] loss: 0.124
    training accuracy :  96.34375
    [43,  6000] loss: 0.129
    training accuracy :  96.26666666666667
    [43,  8000] loss: 0.118
    training accuracy :  96.20625
    [43, 10000] loss: 0.122
    training accuracy :  96.185
    [43, 12000] loss: 0.132
    training accuracy :  96.14166666666667
    [44,  2000] loss: 0.108
    training accuracy :  96.525
    [44,  4000] loss: 0.117
    training accuracy :  96.3375
    [44,  6000] loss: 0.119
    training accuracy :  96.32916666666667
    [44,  8000] loss: 0.122
    training accuracy :  96.246875
    [44, 10000] loss: 0.110
    training accuracy :  96.275
    [44, 12000] loss: 0.125
    training accuracy :  96.20208333333333
    [45,  2000] loss: 0.096
    training accuracy :  96.8375
    [45,  4000] loss: 0.106
    training accuracy :  96.6375
    [45,  6000] loss: 0.115
    training accuracy :  96.49166666666666
    [45,  8000] loss: 0.116
    training accuracy :  96.396875
    [45, 10000] loss: 0.116
    training accuracy :  96.3575
    [45, 12000] loss: 0.111
    training accuracy :  96.34583333333333
    [46,  2000] loss: 0.106
    training accuracy :  96.6375
    [46,  4000] loss: 0.100
    training accuracy :  96.625
    [46,  6000] loss: 0.114
    training accuracy :  96.5875
    [46,  8000] loss: 0.105
    training accuracy :  96.59375
    [46, 10000] loss: 0.104
    training accuracy :  96.61
    [46, 12000] loss: 0.103
    training accuracy :  96.625
    [47,  2000] loss: 0.105
    training accuracy :  96.9875
    [47,  4000] loss: 0.108
    training accuracy :  96.70625
    [47,  6000] loss: 0.108
    training accuracy :  96.625
    [47,  8000] loss: 0.109
    training accuracy :  96.60625
    [47, 10000] loss: 0.115
    training accuracy :  96.555
    [47, 12000] loss: 0.118
    training accuracy :  96.48125
    [48,  2000] loss: 0.087
    training accuracy :  96.9625
    [48,  4000] loss: 0.103
    training accuracy :  96.75625
    [48,  6000] loss: 0.090
    training accuracy :  96.93333333333334
    [48,  8000] loss: 0.105
    training accuracy :  96.81875
    [48, 10000] loss: 0.110
    training accuracy :  96.765
    [48, 12000] loss: 0.105
    training accuracy :  96.74166666666666
    [49,  2000] loss: 0.080
    training accuracy :  97.55
    [49,  4000] loss: 0.109
    training accuracy :  97.00625
    [49,  6000] loss: 0.086
    training accuracy :  97.125
    [49,  8000] loss: 0.105
    training accuracy :  97.015625
    [49, 10000] loss: 0.089
    training accuracy :  97.0225
    [49, 12000] loss: 0.108
    training accuracy :  96.9625
    [50,  2000] loss: 0.085
    training accuracy :  97.4125
    [50,  4000] loss: 0.089
    training accuracy :  97.3
    [50,  6000] loss: 0.096
    training accuracy :  97.16666666666667
    [50,  8000] loss: 0.099
    training accuracy :  97.096875
    [50, 10000] loss: 0.085
    training accuracy :  97.125
    [50, 12000] loss: 0.092
    training accuracy :  97.09791666666666
    Finished Training
    


```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```


```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```


![png](./best_attempt%28model7%29_10_0.png)


    GroundTruth:    cat  ship  ship plane
    


```python
PATH = './cifar_net.pth'
net = Net()
net.to(device)
nwt=net.to(device)
net.load_state_dict(torch.load(PATH))
```




    <All keys matched successfully>




```python
outputs = net(images.to(device))
```


```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```

    Predicted:    cat  ship   car plane
    


```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

    Accuracy of the network on the 10000 test images: 72 %
    

## Attempt 1 (default code):
### accuracy = 53%

## Attempt 2 :
added transforms to image : horizontal flip, vertical flip, rotation by 45 deg
(obseverd that accuracy of the car class was 70%)
###accuracy 47% ; min loss 1.503

##Attempt 3:
added one more fully connected layer in the middle

accuracy of class car reduced to 38% ; class truck increased to 71%; ship increased to 64%

###Accuracy 46% ; min loss 1.511

##Attempt 4:
increased epochs to 25
added train accuracy paramenter
###train accuracy 63.29%;test Accuracy 58% ; loss 1.041

##Attempt 5:
25 epochs with lr 0.001 then 22 epochs with lr 0.0001
###train accuracy 71.69% ; loss 0.800; test accuracy 61%

##Attempt 6:
50 epochs 
increased the number of nodes.
added dropout layer
###train accuracy 92.01; min loss : 0.249; test accuracy 69%

##Attempt 7:
50 epochs
Increased number of nodes even more
###Train accuracy: 97.09%; Test accuracy 72%; loss:0.092

##Attempt 8:
50 epochs
added one more Conv2d layer, also increased the number of nodes slightly
###Train accuracy 95.38 ; Test accuracy 71% ; loss 0.150


##Best attempt = attempt 7



```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
# avg=0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
#     avg+=100 * class_correct[i] / class_total[i]
# avg=avg/10

# print("average of 10 classes = ", avg)
```

    Accuracy of plane : 75 %
    Accuracy of   car : 82 %
    Accuracy of  bird : 64 %
    Accuracy of   cat : 58 %
    Accuracy of  deer : 66 %
    Accuracy of   dog : 63 %
    Accuracy of  frog : 79 %
    Accuracy of horse : 75 %
    Accuracy of  ship : 84 %
    Accuracy of truck : 80 %
    


```python
from matplotlib import pyplot as plt

trainac=[63.29,71.69,92.01,97.09,95.38]
x2=[4,5,6,7,8]
plt.ylabel('training accuracy %')
plt.xlabel('values not recorded for attempt 0-3')
plt.title('training accuracy % for each model(0-3 missing)')
plt.plot(x2,trainac)
plt.show()

x=[1,2,3,4,5,6,7,8]
testacc=[53,47,46,58,61,69,72,71]
plt.xlabel('Attempt no.')
plt.ylabel('Test Accuracy %')
plt.title("Comparision of Test accuracy % of each attempt")
plt.plot(x,testacc)
plt.show()

losses=[1.278,1.503,1.511,1.041,0.800,0.249,0.092,0.150]
plt.xlabel('training loss')
plt.ylabel('attempt no.')
plt.title('training loss recorded for each model')
plt.plot(x,losses)
plt.show()


classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
acc=[75,82,64,58,66,63,97,85,84,80]
plt.bar(classes,acc)
plt.title('Accuracy % of each class for model 7')
plt.ylabel('Accuracy %')
plt.show()

```


![png](./best_attempt%28model7%29_17_0.png)



![png](./best_attempt%28model7%29_17_1.png)



![png](./best_attempt%28model7%29_17_2.png)



![png](./best_attempt%28model7%29_17_3.png)



```python

```
