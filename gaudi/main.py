from torch import optim
from torch.utils.data import Dataset, DataLoader
from config import *
from dataset import *
from loss import *
from model import *
from train import *
from utils import *
import urllib
import zipfile

with open('dl.txt') as f:
    l = f.readlines()
    l = [k.strip() for k in  l]
    l = l[:-1]

urllib.urlretrieve(l[0], "tran.zip")
urllib.urlretrieve(l[0], "valid.zip")

with zipfile.ZipFile('./gaudi/train.zip', 'r') as zip_ref:
    zip_ref.extractall('./gaudi/data/train/')

with zipfile.ZipFile('./gaudi/valid.zip', 'r') as zip_ref:
    zip_ref.extractall('./gaudi/data/valid/')


dataset = MyImageFolder(root_dir="/gaudi/data/")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
gen = Generator(in_channels=3).to(DEVICE)
disc = Discriminator(in_channels=3).to(DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()

if LOAD_MODEL:
    load_checkpoint(
        CHECKPOINT_GEN,
        gen,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
    )

for epoch in range(NUM_EPOCHS):
    train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)
    print(f"epoch -> {epoch}" )
    if SAVE_MODEL:
        save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)