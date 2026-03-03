"""
Pre-trained handwriting synthesis model을 사용하여 글자 단위 학습 데이터 생성.
X-rayLaser/pytorch-handwriting-synthesis-toolkit의 Epoch_56 체크포인트 활용.
"""
import sys
import os
import json
import numpy as np
import h5py
import torch
from pathlib import Path

TOOLKIT_PATH = Path(__file__).parent.parent / "tools" / "handwriting-toolkit"
sys.path.insert(0, str(TOOLKIT_PATH))

from handwriting_synthesis.sampling import HandwritingSynthesizer
from handwriting_synthesis.data import transcriptions_to_tensor


def generate_strokes_no_early_stop(model, tokenizer, text, mu, sd, steps=300, bias=1.0):
    """early stopping 없이 stroke 생성 후 attention 기반으로 자름"""
    sentinel = "  "
    full_text = text + sentinel
    c = transcriptions_to_tensor(tokenizer, [full_text])
    c = c.to(model.device)

    batch_size, u, _ = c.shape
    x = model.get_initial_input().unsqueeze(0)
    w = model.get_initial_window(batch_size)
    k = torch.zeros(batch_size, model.gaussian_components, device=model.device, dtype=torch.float32)
    states = model.get_all_initial_states(batch_size)
    hidden1, hidden2, hidden3 = states

    outputs = []
    text_end_pos = len(text)
    done_count = 0

    for t in range(steps):
        x_with_w = torch.cat([x, w], dim=-1)
        h1, hidden1 = model.lstm1(x_with_w, hidden1)
        phi, k = model.window(h1, c, k)
        w = model.window.matmul_3d(phi, c)
        mixture, hidden2, hidden3 = model.compute_mixture(x, h1, w, hidden2, hidden3)
        pi, mu_m, sd_m, ro, eos = [v[0, 0] for v in mixture]
        with torch.no_grad():
            x_new = get_biased_sample(pi, mu_m, sd_m, ro, eos, model.device, bias)
        outputs.append(x_new)
        x = x_new.unsqueeze(0).unsqueeze(0)

        if t > 10:
            attn_peak = phi[0, 0].argmax().item()
            if attn_peak >= text_end_pos:
                done_count += 1
            else:
                done_count = 0
            if done_count >= 5 and x_new[2] > 0.3:
                break

    if not outputs:
        return None

    result = torch.stack(outputs, dim=0).cpu()
    denorm = result.clone()
    denorm[:, 0] = result[:, 0] * sd[0] + mu[0]
    denorm[:, 1] = result[:, 1] * sd[1] + mu[1]
    denorm[:, 2] = (result[:, 2] > 0.5).float()
    return denorm.numpy()


def get_biased_sample(pi, mu, sd, ro, eos, device, bias):
    """bias 적용된 샘플링. mu/sd는 (K*2,) flat shape"""
    K = pi.shape[0]
    mu = mu.view(K, 2)
    sd = sd.view(K, 2)

    temp = max(1.0 - bias, 0.1)
    pi_adjusted = torch.softmax(torch.log(pi + 1e-8) / temp, dim=-1)
    idx = torch.multinomial(pi_adjusted, 1).item()

    mu_x, mu_y = mu[idx, 0], mu[idx, 1]
    sd_x, sd_y = sd[idx, 0] * temp, sd[idx, 1] * temp
    rho = ro[idx]

    z1 = torch.randn(1, device=device).item()
    z2 = torch.randn(1, device=device).item()
    x = mu_x + sd_x * z1
    y = mu_y + sd_y * (rho * z1 + (1 - rho ** 2).clamp(min=1e-6).sqrt() * z2)

    eos_val = 1.0 if torch.rand(1).item() < (eos.item() ** (1.0 / max(temp, 0.3))) else 0.0
    return torch.tensor([x, y, eos_val], device=device)


def generate_dataset(output_path: Path, our_data_path: Path, samples_per_char: int = 30):
    device = torch.device("cpu")
    checkpoint = str(TOOLKIT_PATH / "checkpoints" / "Epoch_56")
    synth = HandwritingSynthesizer.load(checkpoint, device, bias=0)
    model = synth.model
    mu = synth.mu
    sd = synth.sd

    with h5py.File(our_data_path, 'r') as f:
        our_alphabet = f.attrs['alphabet']
        our_mean = f['mean'][:]
        our_std = f['std'][:]

    chars_upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chars_lower = list("abcdefghijklmnopqrstuvwxyz")
    chars_digit = list("0123456789")
    short_words = [
        "I", "A", "OK", "Hi", "Go", "Do", "No", "My", "To", "Me",
        "Mr", "Dr", "Jr", "Sr", "St", "Ed", "Al", "Jo", "Ty", "Bo",
    ]

    all_texts = []
    for c in chars_upper + chars_lower + chars_digit:
        if c in synth.tokenizer.charset:
            all_texts.extend([(c, samples_per_char)])
    for w in short_words:
        if all(c in synth.tokenizer.charset for c in w):
            all_texts.extend([(w, samples_per_char)])

    all_strokes = []
    all_labels = []
    total = sum(count for _, count in all_texts)
    generated = 0

    for text, count in all_texts:
        successes = 0
        attempts = 0
        while successes < count and attempts < count * 3:
            attempts += 1
            strokes = generate_strokes_no_early_stop(
                model, synth.tokenizer, text, mu, sd,
                steps=150 if len(text) <= 2 else 300,
                bias=0.8
            )
            if strokes is None or len(strokes) < 5:
                continue

            normalized = strokes.copy()
            normalized[:, 0] = (strokes[:, 0] - our_mean[0]) / our_std[0]
            normalized[:, 1] = (strokes[:, 1] - our_mean[1]) / our_std[1]

            all_strokes.append(normalized)
            all_labels.append(text)
            successes += 1
            generated += 1

        print(f"  {text!r:5s}: {successes}/{count} generated ({attempts} attempts)")

    print(f"\nTotal: {generated} samples")

    max_len = max(len(s) for s in all_strokes)
    max_len = min(max_len, 300)

    padded = np.zeros((len(all_strokes), max_len, 3), dtype=np.float32)
    lengths = np.zeros(len(all_strokes), dtype=np.int32)

    for i, s in enumerate(all_strokes):
        length = min(len(s), max_len)
        padded[i, :length] = s[:length]
        lengths[i] = length

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('strokes', data=padded)
        f.create_dataset('texts', data=np.array(all_labels, dtype=h5py.string_dtype()))
        f.create_dataset('lengths', data=lengths)
        f.create_dataset('mean', data=our_mean)
        f.create_dataset('std', data=our_std)
        f.attrs['alphabet'] = our_alphabet
        f.attrs['max_len'] = max_len

    print(f"Saved to {output_path}")
    print(f"  Samples: {len(all_strokes)}")
    print(f"  Max length: {max_len}")
    print(f"  Unique texts: {len(set(all_labels))}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    our_data = project_root / "data" / "train_words.h5"
    output = project_root / "data" / "train_chars.h5"

    if not our_data.exists():
        print(f"Error: {our_data} not found")
        sys.exit(1)

    generate_dataset(output, our_data, samples_per_char=30)
