import pyautogui
import time
import numpy as np
import matplotlib.pyplot as plt

# Script para automatizar o preenchimento de dados para criação de foco no CIVA.
print("Início do script")
time.sleep(2)

# Importando dados dos raios e dos ângulo dos focos:
# r_vec = np.load("../data/focus_dist_vec.npy", allow_pickle=True) # Transformando para mm
# theta_vec = np.load("../data/focus_ang_deg_vec.npy", allow_pickle=True)

angSpan = np.deg2rad(np.arange(-40, 40+.5, .5))
rSpan = 50 * np.ones_like(angSpan)

# Posição do centro do transdutor:
x0 = 0
y0 = 0
z0 = 0
# Inicia as coordenadas dos focus:
x_focus = np.zeros_like(angSpan) + x0
y_focus = np.zeros_like(angSpan) + y0
z_focus = np.zeros_like(angSpan) + z0
# Calcula coordenadas dos focus:
x_focus = np.sin(angSpan) * rSpan
# y_focus = 0
z_focus = np.cos(angSpan) * rSpan

timeWait = .05

nFocus = len(x_focus)
i = 0
print("Contagem regressiva:")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
# Obter posição do ponteiro do mouse: pyautogui.position()
for x, y, z in zip(x_focus, y_focus, z_focus):
    if i == 0:
        pyautogui.click(670, 355)  # Insere X
        pyautogui.press('backspace')
        pyautogui.write(str(nFocus))
        pyautogui.press('enter')
        time.sleep(0.01)

    x_approx = np.round(x, 2)
    y_approx = np.round(y, 2)
    z_approx = np.round(z, 2)

    x_str = str(x_approx)
    y_str = str(y_approx)
    z_str = str(z_approx)

    pyautogui.click(670, 508)  # Insere X
    pyautogui.press('backspace')
    time.sleep(timeWait)
    pyautogui.write(x_str)
    time.sleep(timeWait)
    pyautogui.press('enter')
    time.sleep(timeWait)

    pyautogui.click(670, 528)  # Insere Y
    pyautogui.press('backspace')
    time.sleep(timeWait)
    pyautogui.write(y_str)
    time.sleep(timeWait)
    pyautogui.press('enter')
    time.sleep(timeWait)

    pyautogui.click(670, 547)  # Insere Z
    pyautogui.press('backspace')
    time.sleep(timeWait)
    pyautogui.write(z_str)
    time.sleep(timeWait)
    pyautogui.press('enter')
    time.sleep(timeWait)

    pyautogui.click(705, 372) # Move-se para ir para o próximo ponto
    time.sleep(timeWait)

    i += 1




print("Fim")
