import torch
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds

# Assume device is "cuda" if available, else "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a synthetic point cloud (you can replace this with your own data)
num_points = 3000
points = torch.rand(num_points, 3, device=device)

# Define the point cloud colors and sizes (optional)
colors = torch.full([num_points, 3], 0.5, device=device)
sizes = torch.full([num_points], 0.05, device=device)

# Create a Pointclouds object
point_cloud = Pointclouds(points=[points], features=[colors])

# Define the renderer
R, T = look_at_view_transform(2.7, 0, 0)  # Camera position
cameras = FoVOrthographicCameras(device=device, R=R, T=T)
raster_settings = PointsRasterizationSettings(image_size=512)
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

# Render the point cloud
images = renderer(point_cloud)

# Convert the rendered image to numpy and display it
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.show()
