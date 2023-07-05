from framework import file_mat
from matplotlib import pyplot as plt

data = file_mat.read("../data/DadosEnsaio.mat")

plt.imshow(data.ascan_data[:, 0, 0, :], aspect='auto')
plt.show()
