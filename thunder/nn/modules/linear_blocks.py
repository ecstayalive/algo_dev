import math
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from thunder.nn.mapping import ACTIVATION_CLS_NAME

from .activation import Sin

__all__ = ["LinearBlock", "SirenBlock", "PcLinearBlock"]


class LinearBlock(nn.Module):
    """Multi-Layer Perception Block

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        hidden_features: For example, one tuple (256, 126, 10) stands that
            there are three hidden layer, which sizes are 256, 126, 10.
        activation: The type of activation function used in
            this mlp block
        activation_output: Whether the output needs to be activated


    Note: Use orthogonal method to initialize the linear weight.
        For details: https://arxiv.org/pdf/1609.07093.pdf and
        https://arxiv.org/pdf/math-ph/0609050.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterator[int] = None,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # get block attributions
        activation_fn_name = ACTIVATION_CLS_NAME[activation.lower()]
        activation_cls = getattr(nn, activation_fn_name)
        if hidden_features is not None:
            arch = (in_features, *hidden_features, out_features)
        else:
            arch = (in_features, out_features)
        # create linear block
        layers = []
        for in_dimension, out_dimension in zip(arch[:-2], arch[1:-1]):
            layers.extend(
                (
                    nn.Linear(in_dimension, out_dimension, **factory_kwargs),
                    activation_cls(),
                )
            )
        layers.append(nn.Linear(arch[-2], arch[-1], **factory_kwargs))
        if activate_output:
            layers.append(activation_cls())
        self.linear_block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for layer in self.linear_block:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(gain))

    def forward(self, input: Tensor) -> Tensor:
        return self.linear_block(input)


class SirenBlock(nn.Module):
    """Siren is introduced in "Implicit Neural Representations with Periodic Activation Functions "
    For detail: https://arxiv.org/abs/2006.09661
    Args:
        in_features:
        out_features:
        hidden_features:
        omega:
        device:
        dtype:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterator[int],
        activate_output: bool = False,
        omega: float = 30.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # get block attributions
        activation_cls = Sin
        self.arch = (in_features, *hidden_features, out_features)
        # First layer is very special
        self.siren_head_weight = nn.Parameter(
            torch.empty((self.arch[1], self.arch[0]), **factory_kwargs)
        )
        self.siren_head_bias = nn.Parameter(torch.empty(self.arch[1], **factory_kwargs))
        self.siren_head_out_features = self.arch[1]
        # Other layers
        tail_layers = []
        for in_dimension, out_dimension in zip(self.arch[1:-1], self.arch[2:]):
            tail_layers.extend(
                (
                    activation_cls(),
                    nn.Linear(in_dimension, out_dimension, **factory_kwargs),
                )
            )
        if activate_output:
            tail_layers.append(activation_cls())
        self.siren_tail = nn.Sequential(*tail_layers)
        self.omega = omega
        self.reset_parameters()

    def reset_parameters(self, c: float = 6):
        """For details: https://arxiv.org/abs/2006.09661"""
        nn.init.uniform_(
            self.siren_head_weight,
            -1 / self.siren_head_out_features,
            1 / self.siren_head_out_features,
        )
        nn.init.uniform_(
            self.siren_head_bias,
            -1 / self.siren_head_out_features,
            1 / self.siren_head_out_features,
        )
        for layer in self.siren_tail:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(
                    layer.weight,
                    -math.sqrt(c / layer.out_features),
                    math.sqrt(c / layer.out_features),
                )
                nn.init.uniform_(
                    layer.bias,
                    -math.sqrt(c / layer.out_features),
                    math.sqrt(c / layer.out_features),
                )

    def forward(self, input: Tensor) -> Tensor:
        input = F.linear(input, self.omega * self.siren_head_weight, self.siren_head_bias)
        return self.siren_tail(input)

    def extra_repr(self) -> str:
        return f"(siren_head): Linear(in_features={self.arch[0]}, out_features={self.arch[1]}, bias=True)"


class PcLinear(nn.Module):
    """Predictive Coding Linear Layer
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        device: The device to place tensors on.
        dtype: The data type for tensors.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        nn.init.orthogonal_(self.weight, math.sqrt(gain))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight)


class PcLinearBlock(nn.Module):
    """Modular Predictive Coding Linear Block

    Args:
        in_features: Size of each input sample.
        out_features: Size of the final output/representation layer.
        hidden_features: An iterator of integers defining the sizes of the hidden layers.
            For example, (256, 128) creates two hidden layers with 256 and 128 neurons.
        activation: The type of activation function used for predictions.
        inference_steps: The number of iterative steps to run for inference (updating states).
        inference_lr: The learning rate for updating the states (r) during inference.
        tied_weights: If True, feedback weights are the transpose of feedforward weights.
            This is a common simplification in PC models.
        device: The device to place tensors on.
        dtype: The data type for tensors.

    Note:
        This module's forward pass performs an inference loop to settle the neuron states
        and computes a total prediction error. This error can be used as a loss for an
        external optimizer to update the network weights via backpropagation, which
        simulates the weight update rule in predictive coding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterator[int] = None,
        activation: str = "Tanh",
        inference_steps: int = 20,
        inference_lr: float = 0.1,
        tied_weights: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.tied_weights = tied_weights

        if hidden_features is not None:
            self.arch = (in_features, *hidden_features, out_features)
        else:
            self.arch = (in_features, out_features)
        self.num_layers = len(self.arch)

        try:
            activation_cls = getattr(nn, activation)
            self.activation_fn = activation_cls()
        except AttributeError:
            raise ValueError(f"Activation function '{activation}' not found in torch.nn")

        self.forward_layers = nn.ModuleList()
        self.feedback_layers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.forward_layers.append(nn.Linear(self.arch[i], self.arch[i + 1], **factory_kwargs))
            if not self.tied_weights:
                self.feedback_layers.append(
                    nn.Linear(self.arch[i + 1], self.arch[i], **factory_kwargs)
                )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        """Initialize the weights of the network."""
        for layer in self.forward_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(gain))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if not self.tied_weights:
            for layer in self.feedback_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(gain))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the predictive coding inference process.

        Args:
            x: Input tensor of shape (batch_size, in_features). This is treated
               as the state of the lowest layer (r_0), which is fixed.

        Returns:
            A tuple containing:
            - output (torch.Tensor): The state of the final layer after inference.
            - total_error (torch.Tensor): The sum of squared prediction errors across
              all layers, which can be used as the loss for training.
        """
        batch_size = x.shape[0]

        # --- 1. Initialization ---
        # Initialize states (r) with a single forward pass for a good starting point
        r: List[torch.Tensor] = [x]
        with torch.no_grad():
            temp_r = x
            for layer in self.forward_layers:
                temp_r = self.activation_fn(layer(temp_r))
                r.append(temp_r.clone())

        # Initialize errors (e) to zero
        errors: List[torch.Tensor] = [
            torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype) for dim in self.arch
        ]

        # --- 2. Inference Loop ---
        # Iteratively update states (r) to minimize prediction errors (e)
        for step in range(self.inference_steps):
            # --- 2a. Calculate Prediction Errors (Top-down) ---
            # The top-most layer has no prediction, so its error is its own state
            errors[-1] = r[-1]

            # For other layers, error is (actual state - predicted state)
            for i in range(self.num_layers - 2, -1, -1):
                if self.tied_weights:
                    # Use transposed forward weights for prediction
                    prediction = F.linear(r[i + 1], self.forward_layers[i].weight.t())
                else:
                    prediction = self.feedback_layers[i](r[i + 1])

                errors[i] = r[i] - self.activation_fn(prediction)

            # --- 2b. Update States (Bottom-up Correction) ---
            # Update each layer's state based on its own error and the error it causes below
            for i in range(1, self.num_layers):
                # Gradient from the error at the current layer (bottom-up signal)
                bottom_up_grad = errors[i]

                # Gradient from the error at the layer below (top-down signal)
                # This is the back-propagated error signal
                top_down_grad = F.linear(errors[i - 1], self.forward_layers[i - 1].weight)

                # We need to compute derivative of activation for backprop,
                # but a simpler common update is to just use the error signal.
                # Here we use a standard update rule: dr/dt = -e_i + W^T * e_{i-1}
                # (signs depend on energy function definition)
                # Let's use: dr_i = lr * (-e_i + W_{i-1}^T e_{i-1})

                # Propagate error from layer below back up to the current layer
                feedback_error = F.linear(errors[i - 1], self.forward_layers[i - 1].weight)

                # Update state r_i using gradient descent to minimize errors
                delta_r = -errors[i] + feedback_error
                r[i] = r[i] + self.inference_lr * delta_r

            # The input layer r[0] is clamped to the input 'x' and not updated.
            r[0] = x

        # --- 3. Final Loss Calculation ---
        # The total energy/loss is the sum of squared norms of the error signals.
        # We compute this *after* the inference loop, using the final settled errors.
        final_errors: List[torch.Tensor] = []
        final_errors.append(r[-1])  # Top-level error
        for i in range(self.num_layers - 2, -1, -1):
            if self.tied_weights:
                prediction = F.linear(r[i + 1], self.forward_layers[i].weight.t())
            else:
                prediction = self.feedback_layers[i](r[i + 1])
            final_errors.insert(0, r[i] - self.activation_fn(prediction))

        # Sum of squared errors
        total_error = 0.5 * sum([torch.sum(e * e) for e in final_errors])

        return r[-1], total_error

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference to get the output representation without computing loss.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            output (torch.Tensor): The state of the final layer after inference.
        """
        output, _ = self.forward(x)
        return output
