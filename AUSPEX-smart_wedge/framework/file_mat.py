# -*- coding: utf-8 -*-
"""Docstring do módulo ``file_mat``.

Escrever aqui a documentação completa do módulo ``file_mat``.
"""
import numpy as np
import scipy.io
from scipy.io.matlab.mio5_params import mat_struct

from framework.data_types import DataInsp, InspectionParams, SpecimenParams, ProbeParams


def read(filename):
    """Docstring da função ``read()``.

    Pega os dados de um arquivo .mat para preencher a classe DataInsp.
    """
    # Abre o arquivo .mat de configuração e busca a primeira estrutura do MATLAB encontrada.
    scan_data = None
    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    for key, value in mat.items():
        if type(value) is mat_struct:
            scan_data = value
            break

    if (scan_data is None) or not ('CscanData' in scan_data.__dict__):
        raise Exception("Não encontrada nenhuma estrutura ``scanData`` no arquivo " + filename)

    # Busca, em ``scan_data``, os parâmetros relativos ao transdutor e cria uma instância do objeto
    # ``ProbeParams``.
    num_elem = int(1)   # Os arquivos .mat somente armazenam informações de transdutores simples.
    tp = "linear" if num_elem > 1 else "mono"
    dim = float(scan_data.CscanData.ProbeDiameter)
    pitch = int(0)      # Os arquivos .mat somente armazenam informações de transdutores simples.
    freq_transd = float(scan_data.CscanData.Frequency)
    bw_transd = float(scan_data.CscanData.Bandwidth)/float(scan_data.CscanData.Frequency)
    tp_transd = "cossquare"
    probe_params = ProbeParams(tp=tp,
                               num_elem=num_elem,
                               pitch=pitch,
                               dim=dim,
                               freq=freq_transd,
                               bw=bw_transd,
                               pulse_type=tp_transd)

    # Ajusta a coordenada do centro do transdutor para a origem [0, 0, 0]
    probe_params.elem_center[0, 0] = 0.0
    probe_params.elem_center[0, 1] = 0.0
    probe_params.elem_center[0, 2] = 0.0

    # Busca, em ``scan_data``, os parâmetros relativos ao especimen (peça) inspecionado e cria uma
    # instância do objeto ``SpecimenParams``.
    speed_cl = float(scan_data.CscanData.Cl)
    speed_cs = float(scan_data.CscanData.Cs)
    specimen_params = SpecimenParams(cl=speed_cl, cs=speed_cs)

    # Busca, em ``scan_data``, os parâmetros relativos ao processo de inspeção e cria uma instância do
    # objeto ``InspectionParams``.
    # Busca frequência de amostragem.
    sample_time = float(scan_data.timeScale[1]) - float(scan_data.timeScale[0])
    sample_freq = 1.0/sample_time

    # Busca as informações referentes ao *gate*.
    gate_samples = int(scan_data.CscanData.AscanPoints)
    gate_start = float(scan_data.CscanData.TsGate)
    gate_end = float(scan_data.CscanData.TendGate) + sample_time

    # Cria uma instância do objeto ``InspectionParams``.
    type_insp = "contact"   # Os arquivos .mat somente armazenam informações de inspeções por contato.
    inspection_params = InspectionParams(type_insp=type_insp,
                                         type_capt="sweep",
                                         sample_freq=sample_freq,
                                         gate_start=gate_start,
                                         gate_end=gate_end,
                                         gate_samples=gate_samples)

    # Ajusta as posições dos transdutores para cada passo (*step*) de aquisição.
    # Determina a trajetória do transdutor.
    num_shots = scan_data.CscanData.X.size

    # Busca a coordenada do centro do transdutor para cada *shot*.
    point_center_trans = np.zeros((1, 3))
    for i in range(num_shots):
        point_center_trans[0, 0] = scan_data.CscanData.X[i]
        try:
            inspection_params.step_points[i, :] = point_center_trans
        except IndexError:
            inspection_params.step_points = np.concatenate((inspection_params.step_points, point_center_trans))

    # Cria uma instância do objeto ``DataInsp``.
    dados = DataInsp(inspection_params, specimen_params, probe_params)

    # Faz a leitura dos sinais ``A-scan`` diretamente da estrutura ``scan_data``.
    dados.ascan_data[:, 0, 0, :] = scan_data.AscanValues

    return dados
