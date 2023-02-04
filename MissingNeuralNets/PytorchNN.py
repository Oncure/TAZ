import torch
from torch import nn

# class MaskedLinear(nn.Linear):
#     def __init__(self, mask, **kwargs):
#         super().__init__(*mask.shape, **kwargs)
#         self.mask = mask

#     def forward(self, x):
#         return nn.linear(x, self.weight, self.bias)*self.mask

class NN(nn.Module):
    def __init__(self, *layers):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.SoftMax()
        )
    def forward(self, x):
        return self.layers(x)

learn_rate = 1e-3
num_epochs = 3
batch_size = 50
layers = (15,10,3,1)

# Neural Network:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN(*layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# Training:
train_dataset = 'foo'
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for idx, (data, ans) in enumerate(train_loader):
        data = data.to(device)
        ans = ans.to(device)

        # Forward pass:
        guess = model(data)
        loss  = criterion(guess, ans)

        # Backward pass:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing:
test_dataset  = 'bar'
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True)








''