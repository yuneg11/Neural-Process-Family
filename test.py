import torch

from neural_process import models

x_dim = 1  # Should not be changed
y_dim = 2

model_name = "convnp"

if model_name == "cnp":
    model = models.cnp(x_dim=x_dim, y_dim=y_dim, h_dim=128)
elif model_name == "np":
    model = models.np(x_dim=x_dim, y_dim=y_dim, h_dim=128)
elif model_name == "attncnp":
    model = models.attncnp(x_dim=x_dim, y_dim=y_dim, h_dim=128)
elif model_name == "attnnp":
    model = models.attnnp(x_dim=x_dim, y_dim=y_dim, h_dim=128)
elif model_name == "convcnp":
    model = models.convcnp(y_dim=y_dim, xl=False)
elif model_name == "convnp":
    model = models.convnp(y_dim=y_dim, xl=False)
else:
    model = None

x_context = torch.rand(256, 128, x_dim)
y_context = torch.rand(256, 128, y_dim)
x_target = torch.rand(256, 64, x_dim)
y_target = torch.rand(256, 64, y_dim)

if model_name in ["cnp", "attncnp", "convcnp"]:
    mu, sigma = model(x_context, y_context, x_target)
    loss = model.loss(x_context, y_context, x_target, y_target)
elif model_name in ["np", "attnnp", "convnp"]:
    mu, sigma = model(x_context, y_context, x_target, num_latents=10)
    loss = model.loss(x_context, y_context, x_target, y_target, num_latents=20)
else:
    mu, sigma, loss = None, None, None

print(model)
print(mu.shape)
print(sigma.shape)
print(loss)
