import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
#hyperparametere:
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
z_dim = 128
image_dim = 28*28*1
num_epochs = 10

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixedNoise = torch.randn((batch_size, z_dim)).to(device)
disc_optim = optim.Adam(disc.parameters(), lr = lr)
gen_optim = optim.Adam(gen.parameters(), lr = lr)
cirterion = nn.BCELoss()
transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5), (0.5))
])
#log_dir = "C:\\Users\\Tarek\\OneDrive\\Documents\\VS code files"
dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
loader = DataLoader(dataset,  batch_size = batch_size, shuffle = True)
Writer_Fake = SummaryWriter("G:\\VS code files\\fake")
Writer_Real = SummaryWriter("G:\\VS code files\\real")
step = 0

 
for epoch in range(num_epochs):

    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # train discriminator: maz log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = cirterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = cirterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        disc_optim.step()

        # train Generator: ming log(1 - (D(G(z))) -> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = cirterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        gen_optim.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixedNoise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)

                # Normalize images to be in the range [0, 1]
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True, scale_each=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True, scale_each=True)

                Writer_Fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                Writer_Real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
