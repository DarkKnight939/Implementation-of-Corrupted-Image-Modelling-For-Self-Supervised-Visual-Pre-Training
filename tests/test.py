import torch
import torch.nn.functional as F
import unittest
from ..models.generator import Generator
from ..models.enhancer import CIMEnhancer

class TestCIMModels(unittest.TestCase):
    def setUp(self):
        # Configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = 384
        self.n_heads = 6
        self.n_layers = 6
        self.feature_dim = 384
        self.batch_size = 8
        self.image_size = 256

        # Paths to checkpoints
        self.generator_checkpoint_path = "/content/best_generator_respix.ptt"
        self.enhancer_checkpoint_path = "/content/best_enhancer__respix.pt"

        self.generator = Generator(
            device=self.device,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers
        ).to(self.device)

        self.enhancer = CIMEnhancer(
            feature_dim=self.feature_dim
        ).to(self.device)

        self.load_checkpoints()

    def load_checkpoints(self):
        try:
            generator_state = torch.load(self.generator_checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(generator_state)
            print(f"Loaded generator checkpoint from {self.generator_checkpoint_path}.")
        except FileNotFoundError:
            print(f"Generator checkpoint not found at {self.generator_checkpoint_path}. Skipping loading.")

        try:
            enhancer_state = torch.load(self.enhancer_checkpoint_path, map_location=self.device)
            self.enhancer.load_state_dict(enhancer_state)
            print(f"Loaded enhancer checkpoint from {self.enhancer_checkpoint_path}.")
        except FileNotFoundError:
            print(f"Enhancer checkpoint not found at {self.enhancer_checkpoint_path}. Skipping loading.")

    def create_corrupted_image(self, image):
        mask = torch.ones_like(image, device=self.device)
        patch_size = 32
        num_patches = 5

        for _ in range(num_patches):
            x = torch.randint(0, image.shape[2] - patch_size, (1,)).item()
            y = torch.randint(0, image.shape[3] - patch_size, (1,)).item()
            mask[:, :, x:x + patch_size, y:y + patch_size] = 0

        corrupted_image = image * mask
        return corrupted_image

    def test_generator_forward(self):
        sample_input = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size
        ).to(self.device)
        corrupted_images, latent_repr, attention_maps = self.generator(sample_input)
        self.assertEqual(corrupted_images.shape, sample_input.shape)
        self.assertIsInstance(latent_repr, torch.Tensor)
        print("Generator forward pass successful.")

    def test_enhancer_forward(self):
        ground_truth = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size
        ).to(self.device)
        corrupted_images = self.create_corrupted_image(ground_truth)
        enhanced_images, residuals = self.enhancer(ground_truth, corrupted_images, task='respix')

        # Compute MSE between ground truth and enhanced images
        mse_loss = F.mse_loss(enhanced_images, ground_truth)
        print(f"Enhancer forward pass successful. MSE Loss: {mse_loss.item():.6f}")

        self.assertEqual(enhanced_images.shape, ground_truth.shape)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCIMModels)
    unittest.TextTestRunner().run(suite)
