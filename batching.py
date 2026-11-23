"""Keep independent image batches while respecting H3's batch-one forward pass."""
import torch

KEY = "BExplains.RefPacks.H3.ImageBatch"


def _slice(value, index, batch):
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
        return value[index:index + 1]
    return value


def split_image_batch(executor, x, timestep, context, transformer_options=None, **kwargs):
    streams = isinstance(x, (list, tuple))
    primary = x[0] if streams else x
    batch = primary.shape[0]
    if batch == 1:
        return executor(x, timestep, context, transformer_options or {}, **kwargs)
    outputs = []
    for index in range(batch):
        item = [_slice(stream, index, batch) for stream in x] if streams else _slice(x, index, batch)
        options = {k: _slice(v, index, batch) for k, v in (transformer_options or {}).items()}
        # CFG concatenates complete image batches; retain each item's condition label.
        labels = options.get("cond_or_uncond")
        if isinstance(labels, (list, tuple)) and labels and batch % len(labels) == 0:
            options["cond_or_uncond"] = [labels[index // (batch // len(labels))]]
        values = {k: _slice(v, index, batch) for k, v in kwargs.items()}
        # The multimodal payload contains shared references/token tags, not output batches.
        # Copy its dictionary so native per-call caching cannot change the caller's payload.
        if isinstance(values.get("minimax_payload"), dict):
            values["minimax_payload"] = dict(values["minimax_payload"])
        outputs.append(executor(item, _slice(timestep, index, batch), _slice(context, index, batch), options, **values))
    if streams:
        return [torch.cat([output[stream] for output in outputs], dim=0) for stream in range(len(outputs[0]))]
    return torch.cat(outputs, dim=0)


def enable_image_batches(model):
    import comfy.patcher_extension as ext
    result = model.clone()
    wrappers = result.model_options.setdefault("transformer_options", {}).setdefault("wrappers", {})
    wrappers.setdefault(ext.WrappersMP.DIFFUSION_MODEL, {}).pop(KEY, None)
    ext.add_wrapper_with_key(ext.WrappersMP.DIFFUSION_MODEL, KEY, split_image_batch,
                             result.model_options, is_model_options=True)
    return result
