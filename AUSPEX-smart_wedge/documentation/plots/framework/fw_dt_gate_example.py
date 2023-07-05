import numpy as np
import matplotlib.pyplot as plt
from framework import file_mat

# --- Data ---
data = file_mat.read("DadosEnsaio.mat")

t = data.time_grid
ascan = data.ascan_data

# --- Plots ---
# Gráfico
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, ascan[:, 0, 0, 0])
ax.set_xlabel('Tempo (us)')
ax.set_ylabel('Amplitude')


# Texto "Reflexão superfície"
ax.annotate('Reflexão superfície', xy=(2, 0.0025), xytext=(5, 0.0100),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1,
                            headwidth=7),
            )

# Texto "Intervalo desejado"
ax.annotate('', xy=(10, 0), xytext=(16, -0.0100),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1,
                            headwidth=7),
            )
ax.annotate('', xy=(25, 0), xytext=(16, -0.0100),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1,
                            headwidth=7),
            )
ax.text(13, -0.012, 'Intervalo desejado')

plt.savefig('gate_example.pdf', bbox_inches='tight')
plt.savefig('gate_example.png', bbox_inches='tight')

plt.show()
