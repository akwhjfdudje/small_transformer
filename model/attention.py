import os, torch
if os.name == 'nt':
    os.add_dll_directory(os.path.join(torch.__path__[0], 'lib'))
else:
    os.path.join(torch.__path__[0], 'lib')
import bindings as tb

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5  # scaling factor for attention

    def forward(self, Q, K, V):
        """
        Q, K, V: [B, H, T, D_head]
        Returns:
            output: [B, H, T, D_head]
            weights: [B, H, T, T]
        """
        B, H, T, D = Q.shape

        # Compute scores = Q @ K^T / sqrt(d_k)
        # Flatten batch and heads to use batched_matmul kernel
        Q_flat = Q.reshape(B*H, T, D)
        K_flat = K.reshape(B*H, T, D)

        # Transpose K for matmul: [B*H, D, T]
        K_flat_T = K_flat.transpose(1, 2).contiguous()

        # scores_flat: [B*H, T, T]
        scores_flat = tb.batched_matmul(Q_flat, K_flat_T)
        scores_flat = scores_flat / self.scale

        # Apply softmax along last dim
        # Reshape to 2D [B*H*T, T] for tb.softmax if necessary
        scores_2d = scores_flat.reshape(-1, T)
        scores_softmax_2d = tb.softmax(scores_2d)
        scores_softmax = scores_softmax_2d.view(B*H, T, T)

        # Multiply by V
        V_flat = V.reshape(B*H, T, D)
        output_flat = tb.batched_matmul(scores_softmax, V_flat)

        # Reshape back to [B, H, T, D]
        output = output_flat.view(B, H, T, D)
        weights = scores_softmax.view(B, H, T, T)

        return output, weights

