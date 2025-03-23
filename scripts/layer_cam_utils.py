import torch
import numpy as np
from PIL import Image
from torchcam.methods import LayerCAM
import cv2

import matplotlib.pyplot as plt


class LayerCAMProcessor:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize the Layer-CAM Processor.

        Args:
            model (torch.nn.Module): The model to use for Layer-CAM.
            target_layer (str): The name of the target layer for CAM.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.cam_extractor = LayerCAM(self.model, target_layer=self.target_layer)

    def compute_layer_cam_by_torchcam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate the Layer-CAM heatmap for a specific class.

        Args:
            input_tensor (torch.Tensor): The input tensor (1, C, H, W).
            class_idx (int): The target class index.

        Returns:
            np.ndarray: The Layer-CAM heatmap.
        """
        # Ensure that input_tensor requires gradients
        input_tensor = input_tensor.requires_grad_()

        # Disable gradient computation for the model's forward pass (as we only need gradients for the final layer)
        with torch.set_grad_enabled(True):  # Enables gradient calculation
            output = self.model(input_tensor)  # Perform the forward pass
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            cam = self.cam_extractor(class_idx, output)  # Generate the CAM

            if isinstance(cam, list):
                cam = cam[0]  # Extract the first CAM if it's a list

            cam = cam.squeeze(0).cpu().numpy()  # Process the CAM for visualization
            cam -= cam.min()  # Normalize the CAM
            cam /= cam.max() if cam.max() != 0 else 1  # Avoid division by zero

        return cam

    
    def overlay_cam_on_image(self, image, cam, alpha=0.7):
        """
        Overlay the Layer-CAM heatmap onto the original image.

        Args:
            image (PIL.Image): The original input image.
            cam (np.ndarray): The Layer-CAM heatmap.
            alpha (float): The transparency factor for the heatmap overlay.

        Returns:
            PIL.Image: Image with overlayed Layer-CAM.
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlayed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
        overlayed_image = Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))

        return overlayed_image

    def display_and_save_overlay_cam(self, image: Image.Image, overlayed_image, save_path):
        """
        Plot the original image and the Layer-CAM heatmap.

        Args:
            image (Image.Image): The original input image.
            overlayed_image (PIL.Image): Image with overlayed Layer-CAM.
            save_path (str): Path to save the overlayed image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(overlayed_image)
        axes[1].set_title("Layer-CAM")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        overlayed_image.save(save_path)
        print(f"Overlayed image saved to {save_path}")