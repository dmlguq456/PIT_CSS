# Dual-Path Multi-channel Speech Separation (DPMCN)

---

Current version DPMCN is network to **estimate a basic single mask value** of 0 ~ 1.

Base block is **Conformer**.

## Requirements

---

- [ ]  preparing scp file for training and test dataset
- [ ]  **RIR filter generation** utilizing ‘rir_gen/rir_gen_target.py’
    - [ ]  RIR filter generation for target → appropriate linear array and room specifications
    - [ ]  RIR filter generation for noise (TV) and background diffuse noise
- [ ]  correction of ‘rir_load’ function in ‘run_pit_RI.py’ according to RIR filter.

## Training

---

### Conventional Model (Concat. of Magnitude & IPD)

```bash
python run_pit.py --config ./conf/train_n_Conformer_7ch.yaml --gpus "0"
```

### Proposed DPMCN

```bash
python run_pit.py --config ./conf/train_n_Conformer_7ch.yaml --gpus "0"
```

## Inference

---

### Conventional Model with Mask-MVDR

```bash
python separate_MVDR.py --config ./conf/train_n_Conformer_7ch.yaml --dump-dir ./directoroy_you_save --cuda --dump-mask
```

Inference

### Proposed DPMCN with Mask-MVDR

```bash
python separate_MVDR_DPMCN.py --config ./conf/train_n_DPMCN_7ch_v15.yaml --dump-dir ./directoroy_you_save --cuda --dump-mask
```

## Consideration (later)

---

- [ ]  Changing sigmoid function to ReLU function in mask estimation
- [ ]  Input feature from WPE processing
- [ ]  MVDR → MVDR & WPE joint
