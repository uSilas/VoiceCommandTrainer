"""
PASSO 1 — Treina LSTM supervisionada com os labels dos comandos.
VERSÃO SEM LIBROSA — MFCC implementado manualmente com numpy/scipy.

Execute: python 1_treinar_lstm_supervisionada.py
Gera:    lstm_supervisionada.pt
"""

import random
import torch
import torch.nn as nn
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ========================
# SEED — garante reprodutibilidade entre rodadas
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ========================
# CONFIG
# ========================
SR          = 16000
DURACAO     = 2.50
N_MFCC      = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
EPOCHS      = 50
LR          = 1e-3
BATCH_SIZE  = 32
DATASET_PATH = Path("./dataset")

# Parâmetros do MFCC (equivalentes aos defaults do librosa)
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
FMIN        = 0.0
FMAX        = SR / 2  # Nyquist


# ========================
# MODELO (sem alterações)
# ========================

class LSTMSupervisionado(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        """Retorna o hidden state final — usado como feature depois."""
        _, (hn, _) = self.lstm(x)
        return hn[-1]   # (batch, hidden_size)

    def forward(self, x):
        z = self.encode(x)
        z = self.dropout(z)
        return self.classifier(z)


# ========================
# CARREGAMENTO DE ÁUDIO (substitui librosa.load)
# ========================

def carregar_audio(arquivo, sr_destino=SR):
    """
    Lê um WAV e retorna o sinal em float32 normalizado em [-1, 1],
    re-amostrado para sr_destino se necessário.

    Equivalente a: librosa.load(arquivo, sr=SR)
    """
    sr_original, audio = wavfile.read(arquivo)

    # Converte para float32 em [-1, 1] dependendo do dtype original
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)

    # Se estéreo, converte para mono (média dos canais)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Re-amostragem simples por interpolação linear, se necessário
    if sr_original != sr_destino:
        duracao = len(audio) / sr_original
        n_amostras_novo = int(round(duracao * sr_destino))
        x_antigo = np.linspace(0, duracao, num=len(audio), endpoint=False)
        x_novo   = np.linspace(0, duracao, num=n_amostras_novo, endpoint=False)
        audio    = np.interp(x_novo, x_antigo, audio).astype(np.float32)

    return audio, sr_destino


# ========================
# NORMALIZAÇÃO E FIX LENGTH (substitui librosa.util.*)
# ========================

def normalizar_amplitude(audio):
    """
    Normaliza pelo pico máximo absoluto, para [-1, 1].
    Equivalente a: librosa.util.normalize(audio)
    """
    pico = np.max(np.abs(audio))
    if pico > 0:
        audio = audio / pico
    return audio


def ajustar_duracao(audio, tamanho_fixo):
    """
    Corta ou preenche com zeros para um tamanho fixo de amostras.
    Equivalente a: librosa.util.fix_length(audio, size=tamanho_fixo)
    """
    n = len(audio)
    if n > tamanho_fixo:
        return audio[:tamanho_fixo]
    elif n < tamanho_fixo:
        return np.pad(audio, (0, tamanho_fixo - n), mode="constant")
    return audio


# ========================
# MFCC MANUAL (substitui librosa.feature.mfcc)
# ========================
#
# Pipeline padrão de MFCC:
#   1. Pre-emphasis
#   2. Framing (janelas com overlap)
#   3. Janela de Hamming
#   4. FFT -> espectro de potência
#   5. Filtros mel -> espectro mel
#   6. log
#   7. DCT -> coeficientes MFCC
#

def hz_para_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_para_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def construir_filtros_mel(sr, n_fft, n_mels, fmin, fmax):
    """
    Constrói o banco de filtros triangulares na escala mel.
    Retorna matriz (n_mels, n_fft//2 + 1).
    """
    mel_min = hz_para_mel(fmin)
    mel_max = hz_para_mel(fmax)

    pontos_mel = np.linspace(mel_min, mel_max, n_mels + 2)
    pontos_hz  = mel_para_hz(pontos_mel)

    # Converte frequências para índices de bin da FFT
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


# Pré-computa os filtros mel uma única vez (não depende do áudio)
FILTROS_MEL = construir_filtros_mel(SR, N_FFT, N_MELS, FMIN, FMAX)
JANELA_HAMMING = np.hamming(N_FFT)


def calcular_mfcc(audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Calcula os coeficientes MFCC de um sinal de áudio.
    Retorna matriz (n_mfcc, n_frames) — mesmo formato de librosa.feature.mfcc.
    """
    # 1. Pre-emphasis: realça frequências altas
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 2. Framing — divide o sinal em janelas com overlap
    n_amostras = len(audio_pre)
    n_frames = 1 + (n_amostras - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
        # zero-pad se o áudio for menor que uma janela
        audio_pre = np.pad(audio_pre, (0, n_fft - n_amostras), mode="constant")

    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        inicio = i * hop_length
        fim = inicio + n_fft
        trecho = audio_pre[inicio:fim]
        if len(trecho) < n_fft:
            trecho = np.pad(trecho, (0, n_fft - len(trecho)), mode="constant")
        frames[i] = trecho

    # 3. Janela de Hamming
    frames *= JANELA_HAMMING

    # 4. FFT -> espectro de potência
    espectro = np.fft.rfft(frames, n=n_fft, axis=1)
    potencia = (np.abs(espectro) ** 2) / n_fft   # (n_frames, n_fft//2 + 1)

    # 5. Filtros mel -> espectro mel
    espectro_mel = potencia @ FILTROS_MEL.T      # (n_frames, n_mels)
    espectro_mel = np.maximum(espectro_mel, 1e-10)  # evita log(0)

    # 6. log
    log_mel = np.log(espectro_mel)

    # 7. DCT -> MFCC (mantém os primeiros n_mfcc coeficientes)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfcc.T   # (n_mfcc, n_frames) — igual ao librosa


# ========================
# CARREGA ÁUDIOS
# ========================

def carregar_mfcc(arquivo):
    audio, sr = carregar_audio(arquivo, sr_destino=SR)
    audio = normalizar_amplitude(audio)
    audio = ajustar_duracao(audio, int(SR * DURACAO))
    mfcc  = calcular_mfcc(audio, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T   # (frames, n_mfcc)


print("Carregando áudios...")
sequencias, labels_raw = [], []
n_frames_referencia = None

for pasta in sorted(DATASET_PATH.iterdir()):
    if not pasta.is_dir():
        continue
    for arquivo in sorted(pasta.iterdir()):
        if not arquivo.is_file():
            continue
        try:
            seq = carregar_mfcc(arquivo)

            # Garante que todas as sequências tenham o mesmo número de frames
            # (pode variar 1 frame por arredondamento na FFT)
            if n_frames_referencia is None:
                n_frames_referencia = seq.shape[0]
            elif seq.shape[0] != n_frames_referencia:
                if seq.shape[0] > n_frames_referencia:
                    seq = seq[:n_frames_referencia]
                else:
                    pad = n_frames_referencia - seq.shape[0]
                    seq = np.pad(seq, ((0, pad), (0, 0)), mode="constant")

            sequencias.append(seq)
            labels_raw.append(pasta.name)
        except Exception as e:
            print(f"Erro em {arquivo}: {e}")

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels_raw)
num_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}")
print(f"Amostras: {len(sequencias)}")
print(f"Frames por sequência: {n_frames_referencia}\n")

# Tensor
X = torch.tensor(np.stack(sequencias), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# Normalização Z-score
mean = X.mean(dim=(0, 1), keepdim=True)
std  = X.std(dim=(0, 1), keepdim=True) + 1e-8
X = (X - mean) / std

# Split treino/validação
idx = list(range(len(X)))
idx_train, idx_val = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)

X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]


# ========================
# TREINAMENTO (sem alterações)
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

model     = LSTMSupervisionado(N_MFCC, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_fn   = nn.CrossEntropyLoss()

train_ds = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))

gerador_shuffle = torch.Generator()
gerador_shuffle.manual_seed(SEED)

loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gerador_shuffle
)

best_val_acc = 0.0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    # --- treino ---
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # --- validação ---
    model.eval()
    with torch.no_grad():
        logits   = model(X_val.to(device))
        val_loss = loss_fn(logits, y_val.to(device)).item()
        val_acc  = (logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {total_loss/len(loader):.4f} "
              f"| val_acc: {val_acc:.4f}")

print(f"\nMelhor val_acc: {best_val_acc:.4f}")


# ========================
# SALVA
# ========================

torch.save({
    "model_state":  best_state,
    "mean":         mean.cpu(),
    "std":          std.cpu(),
    "hidden_size":  HIDDEN_SIZE,
    "num_layers":   NUM_LAYERS,
    "n_mfcc":       N_MFCC,
    "n_fft":        N_FFT,
    "hop_length":   HOP_LENGTH,
    "n_mels":       N_MELS,
    "classes":      list(le.classes_),
}, "lstm_supervisionada.pt")

print("Modelo salvo em: lstm_supervisionada.pt")