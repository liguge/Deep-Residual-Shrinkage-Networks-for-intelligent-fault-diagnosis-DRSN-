import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        # a = self.residual_function(x),
        # b = self.shortcut(x),
        # c = a+b
        # return c


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  #CW
        # average = x    #CS
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=4):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def rsnet18():
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])


def rsnet34():
    """ return a RSNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3])

if __name__=='__main__':
    model = rsnet18()
    model.eval()
    # print(model)
    input = torch.randn(1,1,1024)
    y = model(input)
    print(y.size())

    
    '''
    RSNet(
  (conv1): Sequential(
    (0): Conv1d(1, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv2_x): Sequential(
    (0): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=64, out_features=64, bias=True)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=64, out_features=64, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=64, out_features=64, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=64, out_features=64, bias=True)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=64, out_features=64, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=64, out_features=64, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential()
    )
  )
  (conv3_x): Sequential(
    (0): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=128, out_features=128, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=(1,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=128, out_features=128, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential(
        (0): Conv1d(64, 128, kernel_size=(1,), stride=(2,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=128, out_features=128, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=128, out_features=128, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential()
    )
  )
  (conv4_x): Sequential(
    (0): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=256, out_features=256, bias=True)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(128, 256, kernel_size=(3,), stride=(2,), padding=(1,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=256, bias=True)
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=256, out_features=256, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential(
        (0): Conv1d(128, 256, kernel_size=(1,), stride=(2,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=256, out_features=256, bias=True)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=256, out_features=256, bias=True)
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=256, out_features=256, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential()
    )
  )
  (conv5_x): Sequential(
    (0): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=512, out_features=512, bias=True)
          (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=512, out_features=512, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(256, 512, kernel_size=(3,), stride=(2,), padding=(1,), bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential(
        (0): Conv1d(256, 512, kernel_size=(1,), stride=(2,), bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (shrinkage): Shrinkage(
        (gap): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=512, out_features=512, bias=True)
          (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=512, out_features=512, bias=True)
          (4): Sigmoid()
        )
      )
      (residual_function): Sequential(
        (0): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Shrinkage(
          (gap): AdaptiveAvgPool1d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
            (4): Sigmoid()
          )
        )
      )
      (shortcut): Sequential()
    )
  )
  (avg_pool): AdaptiveAvgPool1d(output_size=1)
  (fc): Linear(in_features=512, out_features=4, bias=True)
)
'''
