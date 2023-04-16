import torch

batch_size = 2
logit_size = 3

# generate logits
x1 = torch.randn(batch_size, logit_size)
x2 = torch.randn(batch_size, logit_size)

# average logits
v = torch.stack([x1, x2], axis=0)
print(v)
print(v.shape)

p = torch.mean(v, axis=0)
print(p)
print(p.shape)

# normalize logits
# logits = pic / torch.sum(pic, dim=-1)

# Apply softmax to logits to get probabilities
# probs = torch.softmax(logits, dim=1)
# labels = torch.zeros_like(probs)
# labels.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1)

# print(labels)












# import torch

# def label_smoothing(label, alpha):
#     n_classes = label.size(1)
#     smooth_label = (1 - alpha) * label + (alpha / (n_classes - 1)) * torch.ones_like(label)
#     return smooth_label

# # Example usage
# hard_label = torch.tensor([[0, 1, 0, 0, 0]])
# alpha = 0.1
# soft_label = label_smoothing(hard_label, alpha)
# print(soft_label)


# import numpy as np

# def label_smoothing(label, alpha):
#     n_classes = len(label)
#     smooth_label = (1 - alpha) * label + (alpha / (n_classes - 1)) * np.ones(n_classes)
#     return smooth_label

# # Example usage
# hard_label = np.array([0, 1, 0, 0, 0])
# alpha = 0.1
# soft_label = label_smoothing(hard_label, alpha)
# print(soft_label)




# import torch

# # Example logits
# logits = torch.tensor([[0.5, 2.0, -1.0]])

# # Apply softmax to logits to get probabilities
# probs = torch.softmax(logits, dim=1)

# # Convert probabilities to one-hot labels
# labels = torch.zeros_like(probs)
# labels.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1)

# print(labels)







# import torch

# batch_size = 2
# logit_size = 3

# # generate logits
# x1 = torch.randn(batch_size, logit_size)
# x2 = torch.randn(batch_size, logit_size)
# x3 = torch.randn(batch_size, logit_size)

# # average logits
# pic = torch.mean(torch.stack([x1, x2, x3], axis=1), axis=-1)

# # normalize logits
# logits = pic / torch.sum(pic, dim=-1)

# # Apply softmax to logits to get probabilities
# probs = torch.softmax(logits, dim=1)
# labels = torch.zeros_like(probs)
# labels.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1)

# print(labels)
