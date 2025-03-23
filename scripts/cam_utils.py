import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import CAM

class CAMProcessor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None

        # Hook to capture feature maps
        def hook_fn(_, __, output):
            self.feature_maps = output.detach()

        # Register the hook
        self.target_layer.register_forward_hook(hook_fn)

    def compute_custom_cam(self, input_tensor, class_index=None):

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if class_index is None:
            class_index = torch.argmax(output, dim=1).item()

        # Get the weights of the fully connected layer for the target class
        fc_weights = self.model.fc.weight[class_index].detach()

        # Compute the CAM
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32)
        for i, w in enumerate(fc_weights):
            cam += w * self.feature_maps[0, i, :, :]

        # Normalize the CAM
        cam = cam.numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max() if cam.max() != 0 else cam
        
        return cam
    
    def compute_custom_cam_by_torchcam(self, input_tensor, class_index=None):

        self.model.eval()

        # Initialize the CAM method
        cam_extractor = CAM(self.model, target_layer=self.target_layer, fc_layer='fc')

        # Forward pass and extract CAM
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_index is None:
                class_index = torch.argmax(output, dim=1).item()
            cam_list = cam_extractor(class_index, output)
            cam = cam_list[0]  # Extract the first CAM from the list

        # Normalize the CAM
        cam = cam.squeeze().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max() if cam.max() != 0 else cam

        return cam

    def overlay_cam_on_image(self, image, cam, alpha = 0.7, red_threshold = 0.8, yellow_threshold = 0.5):
        """
        Args:
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

        # Apply colormap (JET) to heatmap
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

    def display_and_save_overlay_cam(self, image, overlayed_image, save_path):
        """
        Args:
            image (PIL.Image): Original input image.
            overlayed_image (PIL.Image): Image with overlayed CAM.
            save_path (str): Path to save the overlayed image.

        """
        # Display the original image and the overlayed image side by side
        _, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        # Overlayed image
        ax[1].imshow(overlayed_image)
        ax[1].set_title("CAM")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the overlayed image
        overlayed_image.save(save_path)
        print(f"Overlayed image saved to {save_path}")
