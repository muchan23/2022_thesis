import numpy as np
import matplotlib.pyplot as plt

D_loss_iter = np.load("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy1/D_loss_iter.npy")
G_loss_iter = np.load("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy1/G_loss_iter.npy")
D_loss_epoch = np.load("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy1/D_loss_epoch.npy")
G_loss_epoch = np.load("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy1/G_loss_epoch.npy")

# iteration graph
data_len = D_loss_iter.size
a = np.arange(1, data_len + 1)

# Geenerator loss
fig0, ax = plt.subplots()
ax.plot(a, G_loss_iter)
ax.set_ylim(2,10)
ax.set_xlabel("iteration")
ax.set_ylabel("G_loss")
ax.set_title("Generator loss")
fig0.savefig('/home/yuki_murakami/ドキュメント/Cycle GAN/data/G_loss_iteration.png')
plt.show()

# Discriminator loss
fig1, ax = plt.subplots()
ax.plot(a, D_loss_iter)
ax.set_ylim(0.2,0.8)
ax.set_xlabel("iteration")
ax.set_ylabel("D_loss")
ax.set_title("Discriminator loss")
fig1.savefig('/home/yuki_murakami/ドキュメント/Cycle GAN/data/D_loss_iteration.png')
plt.show()

# epoch graph
data_len = D_loss_epoch.size
b = np.arange(1, data_len + 1)

# Generator loss
fig2, ax = plt.subplots()
ax.plot(b, G_loss_epoch)
ax.set_ylim(2,10)
ax.set_xlabel("epoch")
ax.set_ylabel("G_loss")
ax.set_title("Generator loss")
fig2.savefig('/home/yuki_murakami/ドキュメント/Cycle GAN/data/G_loss_epoch.png')
plt.show()

# Discriminator loss
fig3, ax = plt.subplots()
ax.plot(b, D_loss_epoch)
ax.set_ylim(0.2,0.8)
ax.set_xlabel("epoch")
ax.set_ylabel("D_loss")
ax.set_title("Discriminator loss")
fig3.savefig('/home/yuki_murakami/ドキュメント/Cycle GAN/data/D_loss_epoch.png')
plt.show()
