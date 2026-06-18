"""
PASSO 2 — Extrai features com LSTM supervisionada + chroma + onset.
VERSÃO SEM LIBROSA — todas as features reimplementadas com numpy/scipy.

Execute: python 2_extracao_features_final.py
Gera:    features_audio_final.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy.fftpack import dct
from pathlib import Path

# ========================
# CONFIG
# ========================
SR           = 16000
DURACAO      = 2.50
N_MFCC       = 20
DATASET_PATH = Path("./dataset")
ENCODER_PATH = "lstm_supervisionada.pt"

N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
FMIN        = 0.0
FMAX        = SR / 2
N_CHROMA    = 12


# ========================
# CARREGA ENCODER LSTM
# ========================

class LSTMSupervisionado(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )
        self.dropout    = nn.Dropout(0.0)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        _, (hn, _) = self.lstm(x)
        return hn[-1]

    def forward(self, x):
        return self.classifier(self.dropout(self.encode(x)))


ckpt        = torch.load(ENCODER_PATH, map_location="cpu", weights_only=False)
HIDDEN_SIZE = ckpt["hidden_size"]
NUM_LAYERS  = ckpt["num_layers"]
num_classes = len(ckpt["classes"])
norm_mean   = ckpt["mean"]
norm_std    = ckpt["std"]
N_FFT       = ckpt.get("n_fft", N_FFT)
HOP_LENGTH  = ckpt.get("hop_length", HOP_LENGTH)
N_MELS      = ckpt.get("n_mels", N_MELS)

encoder = LSTMSupervisionado(N_MFCC, HIDDEN_SIZE, NUM_LAYERS, num_classes)
encoder.load_state_dict(ckpt["model_state"])
encoder.eval()

print(f"Encoder carregado → {HIDDEN_SIZE} dims | classes: {ckpt['classes']}\n")


# ========================
# CARREGAMENTO / NORMALIZAÇÃO DE ÁUDIO
# (idêntico ao script 1)
# ========================

def carregar_audio(arquivo, sr_destino=SR):
    sr_original, audio = wavfile.read(arquivo)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr_original != sr_destino:
        duracao = len(audio) / sr_original
        n_amostras_novo = int(round(duracao * sr_destino))
        x_antigo = np.linspace(0, duracao, num=len(audio), endpoint=False)
        x_novo   = np.linspace(0, duracao, num=n_amostras_novo, endpoint=False)
        audio    = np.interp(x_novo, x_antigo, audio).astype(np.float32)

    return audio, sr_destino


def normalizar_amplitude(audio):
    pico = np.max(np.abs(audio))
    if pico > 0:
        audio = audio / pico
    return audio


def ajustar_duracao(audio, tamanho_fixo):
    n = len(audio)
    if n > tamanho_fixo:
        return audio[:tamanho_fixo]
    elif n < tamanho_fixo:
        return np.pad(audio, (0, tamanho_fixo - n), mode="constant")
    return audio


# ========================
# FRAMING (compartilhado por várias features)
# ========================

JANELA_HAMMING = np.hamming(N_FFT)


def enquadrar(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Divide o sinal em frames com overlap e aplica janela de Hamming.
    Retorna (frames, n_frames) onde frames tem shape (n_frames, n_fft).
    """
    audio_pad = audio
    n_amostras = len(audio_pad)
    n_frames = 1 + (n_amostras - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
        audio_pad = np.pad(audio_pad, (0, n_fft - n_amostras), mode="constant")

    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        inicio = i * hop_length
        fim = inicio + n_fft
        trecho = audio_pad[inicio:fim]
        if len(trecho) < n_fft:
            trecho = np.pad(trecho, (0, n_fft - len(trecho)), mode="constant")
        frames[i] = trecho

    return frames, n_frames


def espectro_potencia(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """STFT -> espectro de potência. Retorna (n_frames, n_fft//2 + 1)."""
    frames, _ = enquadrar(audio, n_fft, hop_length)
    frames_janelados = frames * JANELA_HAMMING
    espectro = np.fft.rfft(frames_janelados, n=n_fft, axis=1)
    potencia = (np.abs(espectro) ** 2) / n_fft
    return potencia, espectro


# ========================
# MFCC (igual ao script 1)
# ========================

def hz_para_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_para_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def construir_filtros_mel(sr, n_fft, n_mels, fmin, fmax):
    mel_min = hz_para_mel(fmin)
    mel_max = hz_para_mel(fmax)

    pontos_mel = np.linspace(mel_min, mel_max, n_mels + 2)
    pontos_hz  = mel_para_hz(pontos_mel)

    bins = np.floor((n_fft + 1) * pontos_hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filtros = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        f_esq, f_centro, f_dir = bins[m - 1], bins[m], bins[m + 1]

        for k in range(f_esq, f_centro):
            if f_centro != f_esq:
                filtros[m - 1, k] = (k - f_esq) / (f_centro - f_esq)

        for k in range(f_centro, f_dir):
            if f_dir != f_centro:
                filtros[m - 1, k] = (f_dir - k) / (f_dir - f_centro)

    return filtros


FILTROS_MEL = construir_filtros_mel(SR, N_FFT, N_MELS, FMIN, FMAX)


def calcular_mfcc(audio, n_mfcc=N_MFCC):
    """Retorna (n_mfcc, n_frames)."""
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    potencia, _ = espectro_potencia(audio_pre)

    espectro_mel = potencia @ FILTROS_MEL.T
    espectro_mel = np.maximum(espectro_mel, 1e-10)
    log_mel = np.log(espectro_mel)

    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc.T   # (n_mfcc, n_frames)


# ========================
# DELTA (substitui librosa.feature.delta)
# ========================

def calcular_delta(feat, largura=9, ordem=1):
    """
    Deriva ao longo do tempo usando regressão de Savitzky-Golay,
    igual ao método padrão do librosa (order=1 ou 2).
    feat: (n_coef, n_frames)
    """
    from scipy.signal import savgol_filter

    n_frames = feat.shape[1]
    # largura precisa ser ímpar e <= n_frames
    largura_ajustada = min(largura, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if largura_ajustada < 3:
        # sequência muito curta: retorna zeros (sem derivada possível)
        return np.zeros_like(feat)

    return savgol_filter(
        feat, window_length=largura_ajustada, polyorder=2,
        deriv=ordem, axis=1, mode="interp"
    )


# ========================
# ZERO CROSSING RATE
# ========================

def calcular_zcr(audio, frame_length=N_FFT, hop_length=HOP_LENGTH):
    """Taxa de cruzamentos por zero, por frame. Retorna (n_frames,)."""
    frames, n_frames = enquadrar(audio, frame_length, hop_length)
    sinais = np.sign(frames)
    sinais[sinais == 0] = 1   # trata zero como positivo (convenção librosa)
    cruzamentos = np.abs(np.diff(sinais, axis=1))
    zcr = np.sum(cruzamentos, axis=1) / (2 * frame_length)
    return zcr


# ========================
# SPECTRAL CENTROID E BANDWIDTH
# ========================

def calcular_spectral_centroid_bandwidth(audio):
    """Retorna (centroid, bandwidth), cada um (n_frames,)."""
    potencia, _ = espectro_potencia(audio)
    magnitude = np.sqrt(potencia)  # espectro de magnitude

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)  # (n_fft//2+1,)

    soma_magnitude = np.sum(magnitude, axis=1) + 1e-10
    centroid = np.sum(magnitude * freqs[np.newaxis, :], axis=1) / soma_magnitude

    # bandwidth = desvio padrão das frequências ponderado pela magnitude
    desvio = freqs[np.newaxis, :] - centroid[:, np.newaxis]
    bandwidth = np.sqrt(
        np.sum(magnitude * (desvio ** 2), axis=1) / soma_magnitude
    )

    return centroid, bandwidth


# ========================
# CHROMA (substitui librosa.feature.chroma_stft)
# ========================

def construir_filtros_chroma(sr, n_fft, n_chroma=N_CHROMA):
    """
    Mapeia bins de frequência para as 12 classes de altura (C, C#, D, ...).
    Retorna matriz (n_chroma, n_fft//2 + 1).
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # evita log(0) no bin DC

    # Frequência -> número de semitom relativo a C
    # 12 * log2(f / f_ref) dá a posição em semitons; mod 12 dá a classe de altura
    f_ref = 16.35160  # frequência de C0 (Hz)
    semitom = 12 * np.log2(freqs / f_ref)
    classe_chroma = np.mod(np.round(semitom), n_chroma).astype(int)

    filtros = np.zeros((n_chroma, n_fft // 2 + 1))
    for k in range(n_fft // 2 + 1):
        filtros[classe_chroma[k], k] += 1.0

    # Normaliza cada filtro pra soma 1 (evita dominância de filtros com mais bins)
    somas = filtros.sum(axis=1, keepdims=True)
    somas[somas == 0] = 1
    filtros = filtros / somas

    return filtros


FILTROS_CHROMA = construir_filtros_chroma(SR, N_FFT, N_CHROMA)

def calcular_chroma(audio):
    """Retorna (n_chroma, n_frames)."""
    potencia, _ = espectro_potencia(audio)
    chroma = potencia @ FILTROS_CHROMA.T   # (n_frames, n_chroma)

    # Normaliza cada frame pelo máximo (convenção librosa: norm='inf' por padrão)
    maximos = np.max(chroma, axis=1, keepdims=True)
    maximos[maximos == 0] = 1
    chroma = chroma / maximos

    return chroma.T   # (n_chroma, n_frames)


# ========================
# ONSET STRENGTH (substitui librosa.onset.onset_strength)
# ========================

def calcular_onset_strength(audio):
    """
    Envelope de força de onset: soma do fluxo espectral positivo
    (aumento de energia entre frames consecutivos), por banda mel.
    Retorna (n_frames,).
    """
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    potencia, _ = espectro_potencia(audio_pre)

    espectro_mel = potencia @ FILTROS_MEL.T
    espectro_mel = np.maximum(espectro_mel, 1e-10)
    log_mel = np.log(espectro_mel)   # (n_frames, n_mels)

    # Fluxo espectral: diferença positiva entre frames consecutivos
    diff = np.diff(log_mel, axis=0)
    diff = np.maximum(diff, 0.0)
    onset_env = np.sum(diff, axis=1)   # (n_frames - 1,)

    # Padroniza tamanho (primeiro frame não tem diferença anterior)
    onset_env = np.concatenate([[0.0], onset_env])

    return onset_env


# ========================
# TEMPO / BEAT TRACK (substitui librosa.beat.beat_track)
# ========================

def calcular_tempo(onset_env, sr=SR, hop_length=HOP_LENGTH):
    """
    Estima o tempo (BPM) via autocorrelação do envelope de onset.
    Retorna um float (BPM).
    """
    if len(onset_env) < 2 or np.all(onset_env == 0):
        return 0.0

    # Remove a média (foca nas variações)
    onset_centrado = onset_env - onset_env.mean()

    autocorr = np.correlate(onset_centrado, onset_centrado, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]   # mantém só lags >= 0

    # Faixa de BPM plausível: 30 a 300
    fps = sr / hop_length   # frames por segundo
    lag_min = int(fps * 60 / 300)   # lag correspondente a 300 BPM
    lag_max = int(fps * 60 / 30)    # lag correspondente a 30 BPM
    lag_max = min(lag_max, len(autocorr) - 1)

    if lag_max <= lag_min:
        return 0.0

    melhor_lag = lag_min + np.argmax(autocorr[lag_min:lag_max + 1])
    if melhor_lag == 0:
        return 0.0

    tempo_bpm = 60.0 * fps / melhor_lag
    return float(tempo_bpm)


# ========================
# ESTATÍSTICAS (igual ao original)
# ========================

def stats(feat):
    """Mean, std, max, min ao longo do tempo para cada coeficiente."""
    return np.concatenate([
        np.mean(feat, axis=1),
        np.std(feat, axis=1),
        np.max(feat, axis=1),
        np.min(feat, axis=1)
    ])


def extrair_lstm(mfcc):
    """Retorna o hidden state da LSTM supervisionada (numpy). mfcc: (n_mfcc, n_frames)."""
    seq = torch.tensor(mfcc.T, dtype=torch.float32)
    seq = (seq - norm_mean.squeeze(0)) / norm_std.squeeze(0)
    seq = seq.unsqueeze(0)
    with torch.no_grad():
        z = encoder.encode(seq)
    return z.squeeze(0).numpy()


# ========================
# EXTRAÇÃO
# ========================

features = []

for pasta in sorted(DATASET_PATH.iterdir()):
    if not pasta.is_dir():
        continue

    label    = pasta.name
    arquivos = sorted(pasta.iterdir())
    print(f"Processando: {label}")

    for i, arquivo in enumerate(arquivos):
        if not arquivo.is_file():
            continue

        try:
            # ---- LOAD ----
            audio, sr = carregar_audio(arquivo, sr_destino=SR)
            audio = normalizar_amplitude(audio)
            audio = ajustar_duracao(audio, int(SR * DURACAO))

            # ---- MFCC + DELTA + DELTA² ----
            mfcc   = calcular_mfcc(audio, n_mfcc=N_MFCC)
            delta  = calcular_delta(mfcc, ordem=1)
            delta2 = calcular_delta(mfcc, ordem=2)

            feat_mfcc = np.concatenate([stats(mfcc), stats(delta), stats(delta2)])

            # ---- FEATURES ESPECTRAIS ----
            energia = np.mean(audio ** 2)
            zcr     = np.mean(calcular_zcr(audio))

            centroid, bandwidth = calcular_spectral_centroid_bandwidth(audio)
            spectral_centroid = np.mean(centroid)
            spectral_bw       = np.mean(bandwidth)

            feat_espectral = np.array([energia, zcr, spectral_centroid, spectral_bw])

            # ---- CHROMA ----
            chroma      = calcular_chroma(audio)
            feat_chroma = stats(chroma)   # 12 coef × 4 stats = 48 dims

            # ---- ONSET STRENGTH + TEMPO ----
            onset_env = calcular_onset_strength(audio)
            tempo     = calcular_tempo(onset_env)

            feat_onset = np.array([
                np.mean(onset_env),
                np.std(onset_env),
                np.max(onset_env),
                tempo
            ])

            # ---- LSTM SUPERVISIONADA ----
            feat_lstm = extrair_lstm(mfcc)

            # ---- CONCATENA TUDO ----
            feat_final = np.concatenate([
                feat_mfcc,        # 240
                feat_espectral,   # 4
                feat_chroma,      # 48
                feat_onset,       # 4
                feat_lstm         # 64
            ])

            # ---- SALVA LINHA ----
            linha = {"arquivo": arquivo.name, "label": label}

            offset = 0
            for j, v in enumerate(feat_mfcc):
                linha[f"f_{j}"] = v
            offset = len(feat_mfcc)

            for j, v in enumerate(feat_espectral):
                linha[f"f_{offset+j}"] = v
            offset += len(feat_espectral)

            for j, v in enumerate(feat_chroma):
                linha[f"chroma_{j}"] = v

            for j, v in enumerate(feat_onset):
                linha[f"onset_{j}"] = v

            for j, v in enumerate(feat_lstm):
                linha[f"lstm_{j}"] = v

            features.append(linha)

            if i % 50 == 0:
                print(f"  {label}: {i}/{len(arquivos)}")

        except Exception as e:
            print(f"  Erro em {arquivo}: {e}")


# ========================
# SALVA CSV
# ========================

df = pd.DataFrame(features)
df.to_csv("features_audio_final.csv", index=False)

n_f      = len([c for c in df.columns if c.startswith("f_")])
n_chroma = len([c for c in df.columns if c.startswith("chroma_")])
n_onset  = len([c for c in df.columns if c.startswith("onset_")])
n_lstm   = len([c for c in df.columns if c.startswith("lstm_")])

print(f"\nSalvo: features_audio_final.csv")
print(f"  Amostras  : {len(df)}")
print(f"  MFCC+esp  : {n_f}")
print(f"  Chroma    : {n_chroma}")
print(f"  Onset     : {n_onset}")
print(f"  LSTM      : {n_lstm}")
print(f"  Total     : {n_f + n_chroma + n_onset + n_lstm} features")