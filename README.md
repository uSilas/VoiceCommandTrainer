# VoiceCommandTrainer

This repository presents an experimental study on speech command recognition for robotic navigation, developed as part of a university course assignment. The challenge proposed by the instructor was to build a voice command classifier capable of handling **noisy audio from diverse sources**, using **small, efficient models** as final classifiers — simulating the constraints of real-world embedded systems.

The core research question guiding this work was:

> *In a real environment with limited hardware and noisy input, how can we build an efficient voice controller?*

The project investigates the performance of **MLP** and **SVM** classifiers using a **collaborative learning approach**, where discriminative audio representations are extracted through a **supervised LSTM network**. Intentionally few audio filters were applied, reflecting the goal of evaluating model robustness under realistic, unclean conditions.

The objective is to classify voice commands such as navigation instructions (e.g., *left, right, forward, back,* and *stop*) from audio recordings and evaluate the effectiveness of different machine learning models on the extracted feature space.

---

## Dataset

The audio dataset used in this study is publicly available at:

🔗 [Google Drive — Audio Dataset](https://drive.google.com/drive/folders/1kojJ3HZkAvbN08krQImQ2ekQtoLPOipA?usp=sharing)

---

## Repository Structure

The pipeline is split into three sequential scripts, named to reflect the execution order:

```text
1_treinar_lstm_supervisionada.py   # Step 1 — Train the supervised LSTM for feature extraction
2_extracao_features_final.py       # Step 2 — Extract and save the feature representations
3_treinar_modelo.ipynb             # Step 3 — Train and evaluate the final MLP/SVM classifiers
```

To replicate the experiments, run the files **in order by filename**. The third file is a Jupyter notebook to allow flexibility during final testing and evaluation.

The trained final model is saved as:

```text
modelo_final.pkl
```

---

## Results

Despite operating under significant constraints — noisy inputs, minimal preprocessing, and lightweight classifiers — the models achieved **high accuracy** and demonstrated promising results in real-world tests. This suggests that the collaborative LSTM + small classifier pipeline is a viable approach even in resource-limited scenarios.

---

## Future Work

The following directions are identified as opportunities to extend this research:

- **Anomaly detection for noise vs. command discrimination** — developing a mechanism to distinguish between actual voice commands and background noise, reducing false activations in uncontrolled environments.

---

## Related Repositories

This repository contains only the **model training pipeline**. The voice-controlled virtual car application that loads and runs the trained model locally was developed separately, across two repositories:

- 🖥️ **Front-end** — Virtual car interface and visualization
  [github.com/uSilas/projeto_reconhecimento_padroes](https://github.com/uSilas/projeto_reconhecimento_padroes)

- ⚙️ **Back-end** — Local inference server that receives audio and returns navigation commands
  [github.com/uSilas/back-end-carwithvoice](https://github.com/uSilas/back-end-carwithvoice)

---

## Application

The developed models are intended for voice-controlled robotic systems, enabling navigation through spoken commands in hardware-constrained environments.
