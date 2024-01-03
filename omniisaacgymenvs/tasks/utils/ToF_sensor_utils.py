import torch



def circle_points(radius, centers, normals, num_points):
    """
    Generate points on a batch of circles in 3D space.

    Args:
    radius (float): The radius of the circles.
    centers (torch.Tensor): a tensor of shape (batch_size, 3) representing the centers of the circles.
    normals (torch.Tensor): a tensor of shape (batch_size, 3) representing the normals to the planes of the circles.
    num_points (int): The number of points to generate on each circle.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, num_points, 3) representing the points on the circles.
    """
    batch_size = centers.shape[0]

    # Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    # Generate random vectors not in the same direction as the normals
    not_normals = torch.rand(batch_size, 3, device='cuda:0')
    while (normals * not_normals).sum(
            dim=-1).max() > 0.99:  # Ensure they're not too similar
        not_normals = torch.rand(batch_size, 3, device='cuda:0')

    # Compute the basis of the planes
    basis1 = torch.cross(normals, not_normals)
    basis1 /= torch.norm(basis1, dim=-1, keepdim=True)
    basis2 = torch.cross(normals, basis1)
    basis2 /= torch.norm(basis2, dim=-1, keepdim=True)

    # Generate points on the circles
    t = torch.arange(0,
                     2 * torch.pi,
                     step=2 * torch.pi / num_points,
                     device='cuda:0')
    circles = centers[:, None, :] + radius * (
        basis1[:, None, :] * torch.cos(t)[None, :, None] +
        basis2[:, None, :] * torch.sin(t)[None, :, None])
    return circles


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a batch of quaternions to rotation matrices.

    Args:
    quaternion (torch.Tensor): a tensor of shape (batch_size, 4) representing the quaternions.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3, 3) representing the rotation matrices.
    """
    w, x, y, z = quaternion.unbind(dim=-1)

    batch_size = quaternion.shape[0]

    rotation_matrix = torch.empty((batch_size, 3, 3), device='cuda:0')

    rotation_matrix[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotation_matrix[:, 0, 1] = 2 * x * y - 2 * z * w
    rotation_matrix[:, 0, 2] = 2 * x * z + 2 * y * w
    rotation_matrix[:, 1, 0] = 2 * x * y + 2 * z * w
    rotation_matrix[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotation_matrix[:, 1, 2] = 2 * y * z - 2 * x * w
    rotation_matrix[:, 2, 0] = 2 * x * z - 2 * y * w
    rotation_matrix[:, 2, 1] = 2 * y * z + 2 * x * w
    rotation_matrix[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

    return rotation_matrix


def find_plane_normal(num_env, quaternions):
    """
    Find the normal to a plane defined by a batch of points and rotations.

    Args:
    num_env: 
    quaternions (torch.Tensor): a tensor of shape (batch_size, 4) representing the rotations.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3) representing the normals to the planes.
    """
    # Convert the quaternions to rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    normals = torch.tensor([1.0, 0.0, 0.0],
                           device='cuda:0').expand(num_env, -1)
    normals = normals.view(num_env, 3, 1)
    rotated_normals = torch.bmm(rotation_matrices, normals)
    return rotated_normals.view(num_env, 3)
