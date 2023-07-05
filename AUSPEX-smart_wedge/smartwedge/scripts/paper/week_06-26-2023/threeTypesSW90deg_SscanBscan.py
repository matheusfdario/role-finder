from framework import file_civa
import numpy as np
import matplotlib.pyplot as plt
from framework.post_proc import envelope


# Script faz a análise usando API do ensaio deslocando o furo para margem da imagem.

def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]


simulations_root = "/media/tekalid/Data/smartwedge_data/06-28-2023/"
original_root = "focus-smartwedge_original-res.civa"  # Diretório da smartwedge original
circle_root = "focus-ponta_adocada-res.civa"  # Diretório da smartwedge com ponta adoçada redonda
square_root = "focus-ponta_adocada_quadrada-res.civa"  # Diretório da smartwedge com ponta adoçada quadrada

betas = np.arange(-40, 40+0.5, .5)

data_original = file_civa.read(simulations_root + original_root, sel_shots=0)
data_circle = file_civa.read(simulations_root + circle_root, sel_shots=0)
data_square = file_civa.read(simulations_root + square_root, sel_shots=0)

t_span_original = data_original.time_grid

# Corta o scan e timegrid para range desejado:
t0 = 0
tend = 100

# Corta o A-scan para limites definidos:
data_original.ascan_data = crop_ascan(data_original.ascan_data, t_span_original, t0, tend)
data_circle.ascan_data = crop_ascan(data_circle.ascan_data, t_span_original, t0, tend)
data_square.ascan_data = crop_ascan(data_square.ascan_data, t_span_original, t0, tend)
t_span = crop_ascan(t_span_original, t_span_original, t0, tend)

#
log_cte = .1
nElement = 80

# Faz a operação de somatório + envoltória:
sscan_original = envelope(np.sum(data_original.ascan_data, axis=2), axis=0)
sscan_original_log = np.log10(sscan_original + log_cte)

sscan_circle = envelope(np.sum(data_circle.ascan_data, axis=2), axis=0)
sscan_circle_log = np.log10(sscan_circle + log_cte)

sscan_square = envelope(np.sum(data_square.ascan_data, axis=2), axis=0)
sscan_square_log = np.log10(sscan_square + log_cte)

bscan_original = envelope(data_original.ascan_data[:, nElement, :, 0], axis=0)
bscan_original_log = np.log10(bscan_original + log_cte)

bscan_circle = envelope(data_circle.ascan_data[:, nElement, :, 0], axis=0)
bscan_circle_log = np.log10(bscan_circle + log_cte)

bscan_square = envelope(data_square.ascan_data[:, nElement, :, 0], axis=0)
bscan_square_log = np.log10(bscan_square + log_cte)


#
vmin_sscan = np.min(sscan_original_log)
vmax_sscan = (np.max(sscan_original_log) + np.min(sscan_original_log))/2
vmin_bscan = np.min(bscan_original_log)
vmax_bscan = np.mean([np.max(bscan_original_log), np.min(bscan_original_log)])

# S-scans dos resultados:
plt.figure(figsize=(12,6))
plt.suptitle("S-Scan para três modelos diferentes de smartwedge com varredura de 90 graus.")
plt.subplot(1, 3, 1)
plt.title("Smartwedge original")
plt.imshow(sscan_original_log, extent=[-40, 40, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_sscan, vmax=vmax_sscan, cmap="magma")
plt.xlabel("Ângulo de varredura na tubulação.")
plt.ylabel(r"Tempo em $\mu$s")

plt.subplot(1, 3, 2)
plt.title("Original com pontas adoçadas")
plt.imshow(sscan_circle_log, extent=[-40, 40, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_sscan, vmax=vmax_sscan, cmap="magma")

plt.subplot(1, 3, 3)
plt.title("Quadrada com pontas adoçadas")
plt.imshow(sscan_square_log, extent=[-40, 40, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_sscan, vmax=vmax_sscan, cmap="magma")

plt.tight_layout()


# B-scans dos resultados:
plt.figure(figsize=(12,6))
plt.suptitle(f"B-Scan (ângulo {betas[nElement]:.1f} de varredura) para três modelos diferentes de smartwedge com varredura "
             f"de 90 graus.")
plt.subplot(1, 3, 1)
plt.title("Smartwedge original")
plt.imshow(bscan_original_log, extent=[0, 64, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_bscan, vmax=vmax_bscan, cmap="magma")
plt.xlabel("Elemento do transdutor")
plt.ylabel(r"Tempo em $\mu$s")

plt.subplot(1, 3, 2)
plt.title("Original com pontas adoçadas")
plt.imshow(bscan_circle_log, extent=[0, 64, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_bscan, vmax=vmax_bscan, cmap="magma")
plt.xlabel("Elemento do transdutor")
plt.ylabel(r"Tempo em $\mu$s")

plt.subplot(1, 3, 3)
plt.title("Quadrada com pontas adoçadas")
plt.imshow(bscan_square_log, extent=[0, 64, t_span[-1][0], t_span[0][0]], aspect='equal', interpolation="None",
           vmin=vmin_bscan, vmax=vmax_bscan, cmap="magma")
plt.xlabel("Elemento do transdutor")
plt.ylabel(r"Tempo em $\mu$s")

plt.tight_layout()