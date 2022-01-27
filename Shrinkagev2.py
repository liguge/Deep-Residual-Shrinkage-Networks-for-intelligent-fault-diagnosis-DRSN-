'''
In order to save memory.
'''

class Shrinkagev2(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkagev2, self).__init__()
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
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        x = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x
