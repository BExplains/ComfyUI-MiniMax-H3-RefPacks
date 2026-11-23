"""H3 visual token accounting."""
import math
from bexplains_refpacks.diagnostics import log, profile_model, SamplingTimings

def reference_stats(blocks, items, latent, text_tokens):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        video = samples.unbind()[0]
    elif isinstance(samples, (tuple, list)):
        video = samples[0]
    else:
        video = samples
    def tokens(z):
        return int(z.shape[2]) * math.ceil(z.shape[-2] / 2) * math.ceil(z.shape[-1] / 2)
    output_tokens = tokens(video)
    total = 0
    lines = []
    for index, (block, item) in enumerate(zip(blocks, items), 1):
        if block['kind'] == 'audio':
            lines.append(f"Reference block {index}: audio, latent {list(block['audio_latent'].shape)}, {block['ref_audio_t']} audio time steps (not included in visual token counts)")
            continue
        z = block["latent"]
        n = tokens(z)
        total += n
        lines.append(f"Reference block {index}: {block['kind']}, latent {list(z.shape)}, {n:,} visual tokens; {item['data'].shape[0]} cached vision frame(s)")
    lines.append(f"Output video grid: {list(video.shape)}; {output_tokens:,} visual tokens")
    lines.append(f"All references: {total:,} visual tokens; output + references: {output_tokens+total:,} (excludes audio/text)")
    lines.append(f"Positive text/vision conditioning: {text_tokens:,} tokens")
    return lines

