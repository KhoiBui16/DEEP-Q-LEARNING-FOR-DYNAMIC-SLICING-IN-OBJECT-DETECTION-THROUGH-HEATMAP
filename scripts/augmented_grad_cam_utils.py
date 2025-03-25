import torch
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from torchcam.methods import GradCAM  # Import GradCAM from torchcam

class AugmentedGradCAMProcessor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute_augmented_grad_cam(self, input_tensor, class_idx=None):
        """
        Computes Augmented Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor.
            class_idx (int, optional): Index of the target class. If None, the predicted class is used.

        Returns:
            numpy.ndarray: Heatmap of Augmented Grad-CAM.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        # Compute Augmented Grad-CAM
        weights = torch.mean(self.gradients.detach(), dim=(2, 3)).squeeze()
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i, :, :].detach()

        cam = torch.relu(cam)
        cam = cam / cam.max() if cam.max() != 0 else torch.zeros_like(cam)

        # Augmentation: Apply Gaussian blur to smooth the heatmap
        cam = cv2.GaussianBlur(cam.cpu().numpy(), (5, 5), 0)

        
        return cam

    def overlay_cam_on_image(self, image, cam, alpha = 0.7, red_threshold = 0.8, yellow_threshold = 0.5):
        """
            image (PIL.Image): The original input image.
            cam (np.ndarray): The Score-CAM heatmap.
            alpha (float): The transparency factor for the original image overlaying.
            red_threshold (float): The threshold for red areas. Regions with values greater than this will show heatmap in red.
            yellow_threshold (float): The threshold for yellow areas. Regions with values between red and yellow will show heatmap in yellow.
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize heatmap to match original image dimensions
        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Normalize heatmap to [0, 255]
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max() if heatmap.max() != 0 else heatmap
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create masks for red, yellow and the remaining regions
        red_mask = heatmap[:, :, 0] > (red_threshold * 255)  # Mask for red regions
        yellow_mask = (heatmap[:, :, 0] > (yellow_threshold * 255)) & (heatmap[:, :, 0] <= (red_threshold * 255))  # Mask for yellow regions
        remaining_mask = ~red_mask & ~yellow_mask  # Mask for the remaining areas (less important regions)

        # Apply heatmap to important regions (red and yellow) with `alpha`
        img[red_mask] = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)[red_mask]  # Apply red regions with `alpha`
        img[yellow_mask] = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)[yellow_mask]  # Apply yellow regions with `alpha`

        # Keep the background intact (remaining areas) with `1 - alpha`
        img[remaining_mask] = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)[remaining_mask]  # Apply background with `1 - alpha`

        # Convert back from BGR to RGB and numpy array to PIL.Image
        overlayed_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return overlayed_image
    
    # def overlay_cam_on_image(self, image, cam, alpha=0.7):
    #     """
    #     Overlay the Augmented Grad-CAM heatmap onto the original image.

    #     Args:
    #         image (PIL.Image): The original input image.
    #         cam (np.ndarray): The Augmented Grad-CAM heatmap.
    #         alpha (float): The transparency factor for the overlay.

    #     Returns:
    #         PIL.Image: Image with overlayed Augmented Grad-CAM.
    #     """
    #     # Convert PIL.Image to numpy array and RGB to BGR
    #     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    #     # Resize heatmap to match original image dimensions
    #     heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))

    #     # Normalize heatmap to [0, 255]
    #     heatmap = np.maximum(heatmap, 0)
    #     heatmap = heatmap / heatmap.max() if heatmap.max() != 0 else heatmap
    #     heatmap = np.uint8(255 * heatmap)

    #     # Apply colormap (JET) to heatmap
    #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    #     # Overlay heatmap onto the original image
    #     overlayed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    #     # Convert back from BGR to RGB and numpy array to PIL.Image
    #     overlayed_image = Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))

    #     return overlayed_image

    def display_and_save_overlay_cam(self, image, overlayed_image, save_path):
        """
        Display the original image and the overlayed image side by side, and save the overlayed image.

        Args:
            image (PIL.Image): Original input image.
            overlayed_image (PIL.Image): Image with overlayed Augmented Grad-CAM.
            save_path (str): Path to save the overlayed image.

        Returns:
            None
        """
        # Display the original image and the overlayed image side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        # Overlayed image
        ax[1].imshow(overlayed_image)
        ax[1].set_title("Augmented Grad-CAM")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the overlayed image
        overlayed_image.save(save_path)
        print(f"Overlayed image saved to {save_path}")