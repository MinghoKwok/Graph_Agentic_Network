# scripts/build_label_vocab.py
import torch

labels = torch.load("data/chameleon/labels.pt")
unique_labels = sorted(set(labels.tolist()))

label_vocab = {label: i for i, label in enumerate(unique_labels)}
inv_label_vocab = {i: label for label, i in label_vocab.items()}

with open("data/chameleon/label_vocab.py", "w") as f:
    f.write("LABELS = {}\n".format(unique_labels))
    f.write("label_vocab = {}\n".format(label_vocab))
    f.write("inv_label_vocab = {}\n".format(inv_label_vocab))
