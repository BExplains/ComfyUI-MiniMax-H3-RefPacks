"""Run with the companion ComfyUI-RefPacks checkout beside this repository."""
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ComfyUI-RefPacks' / 'tests'))
import test_refpacks as h
from test_receivers import setUpModule
import torch


class MiniMaxH3VideoVAE:
    pass


class H3ImageVAE(MiniMaxH3VideoVAE):
    still_frame = 0


class PH3ImageVAE(H3ImageVAE):
    pass


class PH3Tests(unittest.TestCase):
    def vae(self):
        return types.SimpleNamespace(
            first_stage_model=PH3ImageVAE(),
            encode=lambda pixels: torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16),
        )

    def test_detection_including_subclasses(self):
        class DerivedPH3(PH3ImageVAE):
            pass
        for stage in (PH3ImageVAE(), DerivedPH3()):
            self.assertEqual(h.nodes._vae_target(types.SimpleNamespace(first_stage_model=stage)), 'h3_image')

    def test_image_reference_encoding_records_ph3_provenance(self):
        pack = h.nodes.BExplainsH3EncodeImages().encode(self.vae(), torch.zeros(1, 64, 64, 3), 'PH3')[0]
        self.assertEqual(pack['source_target'], 'h3_image')
        self.assertTrue(pack['encoder_class'].endswith('.PH3ImageVAE'))
        self.assertEqual(pack['entries'][0]['latent'].shape[2], 1)

    def test_still_output_on_both_model_architectures(self):
        refs = h.core.add_pack(h.pack(), role='subject')
        for image_model in (True, False):
            vae = self.vae()
            result = h.nodes.BExplainsRefPackToH3().compile(
                h.FakeModel(image_model), h.RecordingClip(), refs, '<Subject 1> smiles', '',
                64, 64, 1, False, vae, False, generation_mode='Image')
            self.assertIn('output intent: h3_image', result[4])
            self.assertIs(result[5], vae)

    def test_video_output_still_requires_video_vae(self):
        refs = h.core.add_pack(h.pack(), role='subject')
        with self.assertRaisesRegex(ValueError, 'Video VAE'):
            h.nodes.BExplainsRefPackToH3().compile(
                h.FakeModel(), h.RecordingClip(), refs, '', '',
                64, 64, 5, False, self.vae(), False, generation_mode='Video')
