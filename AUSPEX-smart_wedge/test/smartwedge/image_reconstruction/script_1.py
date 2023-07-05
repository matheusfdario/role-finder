import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k, file_civa
from framework.post_proc import envelope
from parameter_estimation.intsurf_estimation import img_line
from parameter_estimation import intsurf_estimation

data_date = "16-03-2023"
type_exp = "ensaio_1"
data_root = f"/media/tekalid/Data/smartwedge_data/{data_date}/" + type_exp + "/"
extension = ".m2k"
shot = 0
data = file_m2k.read(data_root + type_exp + extension, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                     tp_transd='gaussian')
data_ref = file_m2k.read(data_root + type_exp + "_referencia" + extension, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                     tp_transd='gaussian')

log_cte = 1500

img = envelope(data.ascan_data_sum, axis=0)
img_log = np.log10(img + log_cte)

img_filt = np.copy(img)
# plt.hist(np.log10(img.flatten() + 100), cumulative=True, bins=1000, density=True)
img = np.copy(img_filt)

img_ref = envelope(data_ref.ascan_data_sum, axis=0)
img_ref_log = np.log10(img_ref + log_cte)

img_sub = envelope(data.ascan_data_sum - data_ref.ascan_data_sum, axis=0)
img_sub[img_sub > 2000] = 2000

t_span = data.time_grid
z_span = t_span[:, 0]*1e-6 * 6300 * 1e3  # in mm
z_span = z_span - z_span.min()

# SEAM ROI:
z0 = 35
zf = 52
gamma_cte = 3.3

# SEAM processing:
amp_func = np.linspace(0, 1, num=img.shape[0])[::-1]
amp_func = amp_func ** gamma_cte
img_grad = np.tile(amp_func, [img.shape[1], 1]).transpose()
img_sub_log = np.log10(img_sub[:, :, 0] + log_cte)
img_sub_log_grad = np.log10(img_sub[:, :, 0] * img_grad + log_cte)

img_roi = img_sub_log_grad[np.argmin(np.power(z_span-z0, 2)) : np.argmin(np.power(z_span-zf, 2))]
z_span_for_seam = z_span[np.argmin(np.power(z_span-z0, 2)): np.argmin(np.power(z_span-zf, 2))]

z_max = z_span[np.argmin(np.power(z_span-z0, 2))] + \
        z_span[np.argmax(img_roi, axis=0)]

# Recorte da superfície interna:
lambda_param = 1e-40
rho_param = 100
# Aplicação do SEAM:
inner_norm_img = img_roi / img_roi.max()
y = inner_norm_img
a = img_line(y)
zeta = z_span_for_seam
z = zeta[a[0].astype(int)]
w = np.diag((a[1]))
print(f"SEAM: Estimando superfíce Interba com SEAM")
z_seam, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                  eta=.999, itmax=10, tol=1e-3)



plt.subplot(2, 3, 1)
plt.suptitle(f"Resultados p/ o {type_exp} feito no dia {data_date} com smartwedge v05 (compacta)")
plt.imshow(img_log, extent=[0, 161, z_span[-1], z_span[0]], aspect='auto', cmap="magma", interpolation="None")
plt.title("(1) Inspeção")
plt.grid()
plt.subplot(2, 3, 2)
plt.imshow(img_ref_log, extent=[0, 161, z_span[-1], z_span[0]], aspect='auto', cmap="magma", interpolation="None")
plt.title("(2) Referência")
plt.grid()
plt.subplot(2, 3, 3)
plt.imshow(img_sub_log, extent=[0, 161, z_span[-1], z_span[0]], aspect='auto', cmap="magma", interpolation="None")
plt.title("(3) Inspeção - Referência após aplicação de filtragem não linear")
plt.grid()

plt.subplot(2, 3, 4)
plt.imshow(img_sub_log_grad, extent=[0, 161, z_span[-1], z_span[0]], aspect='auto', cmap="magma", interpolation="None")
plt.plot(np.arange(0, 161), np.arange(0, 161)*0 + z0, color='g', label='_')
plt.plot(np.arange(0, 161), np.arange(0, 161)*0 + zf, color='g', label="ROI")
plt.plot(np.arange(0, 161), z_max, 'o', markersize=1, color='y', label="Max")
plt.plot(np.arange(0, 161), z_seam, 'o', markersize=1, color='b', label="SEAM")
plt.legend()
plt.title(fr"(4) $\lambda$ = {lambda_param} $\rho$ = {rho_param} com gamma ($\gamma$={gamma_cte})")
plt.grid()

# SEAM processing:
amp_func = np.linspace(0, 1, num=img.shape[0])[::-1]
img_grad = np.tile(amp_func, [img.shape[1], 1]).transpose()
img_sub_log = np.log10(img_sub[:, :, 0] + log_cte)

img_roi = img_sub_log[np.argmin(np.power(z_span-z0, 2)) : np.argmin(np.power(z_span-zf, 2))]
z_span_for_seam = z_span[np.argmin(np.power(z_span-z0, 2)): np.argmin(np.power(z_span-zf, 2))]

z_max = z_span[np.argmin(np.power(z_span-z0, 2))] + \
        z_span[np.argmax(img_roi, axis=0)]
# Recorte da superfície interna:
lambda_param = 1e-40
rho_param = 100
# Aplicação do SEAM:
inner_norm_img = img_roi / img_roi.max()
y = inner_norm_img
a = img_line(y)
zeta = z_span_for_seam
z = zeta[a[0].astype(int)]
w = np.diag((a[1]))
print(f"SEAM: Estimando superfíce Interba com SEAM")
z_seam_2, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                  eta=.999, itmax=10, tol=1e-3)


plt.subplot(2, 3, 5)
plt.imshow(img_sub_log, extent=[0, 161, z_span[-1], z_span[0]], aspect='auto', cmap="magma", interpolation="None")
plt.plot(np.arange(0, 161), np.arange(0, 161)*0 + z0, color='g', label='_')
plt.plot(np.arange(0, 161), np.arange(0, 161)*0 + zf, color='g', label="ROI")
plt.plot(np.arange(0, 161), z_max, 'o', markersize=1, color='y', label="Max without gamma")
plt.plot(np.arange(0, 161), z_seam, 'o', markersize=1, color='b', label="SEAM with gamma")
plt.plot(np.arange(0, 161), z_seam_2, 'o', markersize=1, color='r', label="SEAM without gamma")
plt.legend()
plt.title(fr"(5) $\lambda$ = {lambda_param} $\rho$ = {rho_param} sem gamma")
plt.grid()