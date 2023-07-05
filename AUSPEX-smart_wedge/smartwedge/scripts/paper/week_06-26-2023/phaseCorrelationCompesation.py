from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt
from framework.post_proc import envelope


# Script faz a análise usando API do ensaio deslocando o furo para margem da imagem.

def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]

def _fftpos2coordshift(shift, n):
    # Deslocamentos positivos pertencerão ao intervalo [0, N/2[ onde N é o número de amostras.
    # Deslocamentos negativos pertencerão ao intervalo [N/2, N[ onde N é o número de amostras.
    if shift > n / 2:
        return -(n - shift)
    else:
        return shift

def estimateShiftPerColumn(img1, img2, originalRows=1249):
    from numpy.fft import fftshift, fft, ifft
    import scipy

    if len(img1.shape) == 1:
        rows = img1.shape[0]
        columns = 1
    else:
        rows, columns = img1.shape
    shiftVector = np.zeros(columns, dtype='int')
    for c in range(columns):
        c1 = img1[:, c]
        c2 = img2[:, c]
        #
        if c%20==0:
            plt.title("Algumas colunas do b-scan na ROI")
            plt.plot(c1 + c/4, color=[(columns - c) / columns, 0, 0])
            plt.plot(c2 + c/4, ':', color=[(columns - c) / columns, 0, 0], label=f"Coluna {c+1}")
            plt.legend()
            # shift = np.argmax(np.convolve(c1, c2))
            # plt.plot(c1, color=[(columns - c) / columns, 0, 0])
            # plt.plot(c2, ':', color=[(columns - c) / columns, 0, 0], legend=f"Coluna {}")
            # plt.plot(np.roll(c2, shift), ':', color=[(columns - c) / columns, 0, 1])


        C1 = fftshift(fft(c1))
        C2 = fftshift(fft(c2))
        Q = C1 * np.conj(C2) / np.abs(C1 * np.conj(C2))
        q = np.real(ifft(Q))
        raw_shift = np.argmax(q)  # Máximo do normalized crosspower spectrum
        shift = _fftpos2coordshift(raw_shift, rows)


        # shift = np.argmax(scipy.signal.correlate(c1, c2))
        # if shift > 1000:
        #     shift -= originalRows
        shiftVector[c] = shift
    return shiftVector

def compensateIncidentField(img1, img2, shiftVector):
    if np.max(shiftVector) < 0:
        offset = 0
    else:
        offset = np.max(shiftVector)
    newNumRows = offset + img1.shape[0]
    column = img1.shape[1]
    newExp = np.zeros((newNumRows, column))
    newRef = np.zeros((newNumRows, column))
    subtracted = np.zeros((newNumRows, column))
    newExp[:img1.shape[0], :] = np.copy(img1)
    newRef[:img2.shape[0], :] = np.copy(img2)
    for i, shift in enumerate(shiftVector):
        newRef[:, i] = np.roll(newRef[:, i], shift=shift)
        subtracted[:, i] = newExp[:, i] - newRef[:, i]
    Exp = newExp[:img1.shape[0], :]
    Ref = newRef[:img1.shape[0], :]
    Sub = subtracted[:img1.shape[0], :]
    return Exp, Ref, Sub

# Análise dos ascans:

experiment_root = "/media/tekalid/Data/smartwedge_data/06-26-2023/"
# experiment_ref_1 = "ref_onda_com_foco_submerso.m2k"
# experiment_1 = "onda_com_foco_submerso.m2k"
# experiment_ref = "ref_onda_plana_submerso.m2k"
experiment_1 = "posicao_01.m2k"
experiment_2 = "posicao_02.m2k"
experiment_3 = "posicao_03.m2k"
experiment_4 = "posicao_04.m2k"
experiment_5 = "posicao_05.m2k"
experiment_6 = "posicao_06.m2k"
experiment_ref = "referencia.m2k"
betas = np.linspace(-40, 40, 161)






# New t_span


# Operações
log_cte = .5

j = 1
experiment_name = f"posicao_{j:02d}.m2k"
data_experiment = file_m2k.read(experiment_root + experiment_name, type_insp='contact', water_path=0, freq_transd=5,
                                bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                         bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

t_span_original = data_ref.time_grid

# Corta o scan e timegrid para range desejado:
t0 = 20
tend = 60

# Corta o A-scan para limites definidos:
t_span = crop_ascan(t_span_original, t_span_original, t0, tend)
data_experiment.ascan_data = crop_ascan(data_experiment.ascan_data, t_span_original, t0, tend)
data_ref.ascan_data = crop_ascan(data_ref.ascan_data, t_span_original, t0, tend)

# Faz a operação de somatório + envoltória:
# sscan_exp = envelope(np.sum(data_experiment.ascan_data - data_ref.ascan_data, axis=2), axis=0)
bscan_exp = envelope(data_experiment.ascan_data, axis=0)
bscan_exp_log = np.log10(bscan_exp + log_cte)
bscan_ref = envelope(data_ref.ascan_data, axis=0)
bscan_ref_log = np.log10(bscan_ref + log_cte)
bscan_subtraction = envelope(data_experiment.ascan_data - data_ref.ascan_data, axis=0)
bscan_subtraction_log = np.log10(bscan_subtraction + log_cte)


# Definir mesma colorbar:
vmin_sscan = bscan_exp_log.min()
vmax_sscan = bscan_exp_log.max()


# Plota os dados:
plt.subplot(1, 3, 1)
idxAng = 80
plt.title(f"B-scan do ângulo {betas[idxAng]:.1f}°")
plt.imshow(bscan_exp_log[:, idxAng, :, 0], extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

plt.subplot(1, 3, 2)
plt.title(f"B-scan de referência do ângulo {betas[idxAng]:.1f}°")
plt.imshow(bscan_ref_log[:, idxAng, :, 0], extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

plt.subplot(1, 3, 3)
plt.title(f"B-scan da subtração direta")
plt.imshow(bscan_subtraction_log[:, idxAng, :, 0], extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)


##############################################

shiftVector = estimateShiftPerColumn(bscan_exp_log[700:1950, idxAng, :, 0], bscan_ref_log[700:1950, idxAng, :, 0])
a, b, c = compensateIncidentField(data_experiment.ascan_data[:, idxAng, :, 0], data_ref.ascan_data[:, idxAng, :, 0], shiftVector)
a = np.log10(envelope(a, axis=0) + log_cte)
b = np.log10(envelope(b, axis=0) + log_cte)
c = np.log10(envelope(c, axis=0) + log_cte)


# Plota os dados:
plt.figure()
plt.subplot(1, 3, 1)
idxAng = 80
plt.title(f"B-scan do ângulo {betas[idxAng]:.1f}°")
plt.imshow(bscan_exp_log[:, idxAng, :, 0], extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

plt.subplot(1, 3, 2)
plt.title(f"B-scan de referência do ângulo {betas[idxAng]:.1f}°")
plt.imshow(bscan_ref_log[:, idxAng, :, 0], extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)

plt.subplot(1, 3, 3)
plt.title(f"B-scan da subtração deslocada")
plt.imshow(c, extent=[0, 64, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='equal',
           interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)