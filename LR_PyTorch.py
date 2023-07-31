from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 2 * self.x - 1
        self.y = self.f + 0.5*torch.randn((self.x.shape[0],1))
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = Data()
data = DataLoader(dataset, batch_size = 16, shuffle = True)


class LinearRegression1(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)
    
    def train(self, train_data, epochs, optimizer, loss_function):
        for epoch in range(epochs):
            for x, y in train_data:
                yhat = self(x)
                loss = loss_function(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
class LinearRegression2(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.nn.Module.__init__(self)
        self.w = torch.nn.Parameter(torch.randn(()))
        self.bias = torch.nn.Parameter(torch.randn(()))
        
    def forward(self, x):
        return self.w*x + self.bias
    
    def train(self, train_data, epochs, lr):
        for epoch in range(epochs):
            for x, y in train_data:
                yhat = self(x)
                loss = torch.sum((yhat - y)**2)/x.shape[0]
                loss.backward()
                self.w.data -= lr*self.w.grad
                self.bias.data -= lr*self.bias.grad
                self.w.grad.data.zero_()
                self.bias.grad.data.zero_()
    
class LinearRegression3(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.nn.Module.__init__(self)
        self.w = torch.randn(())
        self.bias = torch.randn(())
        
    def forward(self, x):
        return self.w*x + self.bias
    
    def train(self, train_data, epochs, lr):
        for epoch in range(epochs):
            for x, y in train_data:
                yhat = self(x)
                loss = torch.sum((yhat - y)**2)/x.shape[0]
                grad_w = (2/x.shape[0]) * torch.sum((y - self.w*x - self.bias)*-x)
                grad_b = (2/x.shape[0]) * torch.sum((y - self.w*x - self.bias)*-1)
                
                self.w -= lr*grad_w
                self.bias -= lr*grad_b
                
                
model1 = LinearRegression1(1, 1)
model2 = LinearRegression2(1, 1)
model3 = LinearRegression3(1, 1)

model1.train(data, 50, torch.optim.SGD(model1.parameters(), lr = 0.01), torch.nn.MSELoss())
model2.train(data, 50, 0.01)
model3.train(data, 50, 0.01)


models = [(model1.linear.weight.item(), model1.linear.bias.item()),
          (model2.w.item(), model2.bias.item()),
          (model3.w.item(), model3.bias.item())]


x = dataset.x.numpy()
fig, axs = plt.subplots(3)
for i in range(3):
        w, b = models[i]
        axs[i].plot(x, dataset.y.numpy(), 'rx', label = 'Data points')
        axs[i].plot(x, dataset.f.numpy(), label = 'Original function (y = 2x - 1)' ,alpha = 0.8)
        axs[i].plot(x, [w*j + b for j in x],
                    label = f"Learned function (y = {round(w, 4)}x + {round(b, 4)})")
        axs[i].legend()
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')
    ax.label_outer()
plt.show()

