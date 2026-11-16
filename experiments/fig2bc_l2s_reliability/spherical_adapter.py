"""Input-dependent prototypes using the unchanged native spherical transform."""
import torch
from methods.spherical_steering.steering import spherical_geometric_logic


def get_l2s_spherical_intervention(mlp, *, kappa=20.0, alpha=0.15, beta=0.1, stats=None):
    mlp.eval()
    def hook(output, layer_name=None, start_idx=None):
        del layer_name
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        start = hidden.shape[1]-1 if start_idx is None else max(0,min(int(start_idx),hidden.shape[1]-1))
        device = next(mlp.parameters()).device
        with torch.no_grad():
            predictions = mlp(hidden[:, start:].float().to(device)).to(hidden.device)
        for b in range(hidden.shape[0]):
            for offset in range(predictions.shape[1]):
                vector = predictions[b,offset]
                n = vector.norm()
                triggered = False
                if n > 1e-8:
                    target = vector/n
                    edited[b,start+offset], triggered = spherical_geometric_logic(
                        hidden[b,start+offset],target,-target,kappa,alpha,beta)
                if stats is not None:
                    stats['total'] = stats.get('total',0)+1
                    stats['steered'] = stats.get('steered',0)+int(triggered)
        return (edited,)+output[1:] if isinstance(output,tuple) else edited
    return hook
