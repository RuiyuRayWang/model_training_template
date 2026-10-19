import os
import importlib.util
import torch
from safetensors.torch import load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module {module_name} from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_from_config(cfg):
    model_name = cfg.get('train', {}).get('model', 'DeepMSN')

    if model_name == 'Minuet':
        model_module = load_module_from_path(
            'local_minuet_attn_pooling',
            os.path.join(REPO_ROOT, 'models', 'minuet_attn_pooling.py')
        )
        return model_module.Minuet(config=cfg)

    if model_name == 'DeepMSN':
        model_module = load_module_from_path(
            'local_deepmsn',
            os.path.join(REPO_ROOT, 'models', 'deepmsn.py')
        )
        return model_module.DeepMSN(config=cfg)

    if model_name in {'DeepMSNAttnPooling', 'deepmsn_attn_pooling'}:
        model_module = load_module_from_path(
            'local_deepmsn_attn_pooling',
            os.path.join(REPO_ROOT, 'models', 'deepmsn_attn_pooling.py')
        )
        return model_module.DeepMSNAttnPooling(config=cfg)

    if model_name in {'DeepMSNMeanPooling', 'deepmsn_mean_pooling'}:
        model_module = load_module_from_path(
            'local_deepmsn_mean_pooling',
            os.path.join(REPO_ROOT, 'models', 'deepmsn_mean_pooling.py')
        )
        return model_module.DeepMSNMeanPooling(config=cfg)

    if model_name in {'DeepMSNCnnOnly', 'deepmsn_cnn_only'}:
        model_module = load_module_from_path(
            'local_deepmsn_cnn_only',
            os.path.join(REPO_ROOT, 'models', 'deepmsn_cnn_only.py')
        )
        return model_module.DeepMSNCnnOnly(config=cfg)

    raise ValueError(f'Unsupported train.model: {model_name}')


def load_model_from_safetensors(checkpoint_dir, cfg):
    model = build_model_from_config(cfg)
    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f'SafeTensors file not found at: {safetensors_path}')

    state_dict = load_file(safetensors_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Handle DDP prefixes like "module."
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                stripped[k[len('module.'):]] = v
            else:
                stripped[k] = v
        model.load_state_dict(stripped, strict=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'Loaded model from {safetensors_path}')
    print(f'Using device: {device}')
    return model, device