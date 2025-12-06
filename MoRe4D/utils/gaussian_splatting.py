from math import isqrt

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor


def gs_render(
        intrinsic, 
        extrinsic,  
        image_shape, 
        means, 
        scale,
        rotation,
        color, 
        opacities
    ):
    background_color = torch.Tensor([0.0, 0.0, 0.0]).cuda()
    covariances = build_covariance(
        scale.unsqueeze(0).cuda(), 
        rotation.unsqueeze(0).cuda()
    ).unsqueeze(1).expand(-1, means.shape[0], -1, -1)
    color = render_cuda(
        extrinsics=extrinsic.unsqueeze(0).cuda(),
        intrinsics=intrinsic.unsqueeze(0).cuda(),
        near=torch.Tensor([0.5]).cuda(),
        far=torch.Tensor([1000.0]).cuda(),
        image_shape=image_shape,
        background_color=background_color.unsqueeze(0),
        gaussian_means=means.unsqueeze(0).cuda(),
        gaussian_covariances=covariances,
        gaussian_sh_coefficients=color.unsqueeze(0).unsqueeze(-1).cuda(),
        gaussian_opacities=opacities.unsqueeze(0).cuda(),
        scale_invariant=False,
        use_sh=False
    )

    return color

def gs_render_batch_moving(
        intrinsic, 
        extrinsic,  
        image_shape, 
        means, 
        scale,
        rotation,
        color, 
        opacities
    ):
    B, T, N, _ = means.shape
    background_color = torch.Tensor([0.0, 0.0, 0.0]).cuda()
    covariances = build_covariance(
        scale.unsqueeze(0).cuda(), 
        rotation.unsqueeze(0).cuda()
    ).unsqueeze(1).expand(B*T, N, -1, -1)
    color = render_cuda(
        extrinsics=extrinsic.repeat(B, 1, 1).cuda(),
        intrinsics=intrinsic.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, 3, 3).cuda(),
        near=torch.Tensor([0.5]).expand(B*T).cuda(),
        far=torch.Tensor([1000.0]).expand(B*T).cuda(),
        image_shape=image_shape,
        background_color=background_color.unsqueeze(0).expand(B*T, -1),
        gaussian_means=means.reshape(B*T, N, 3).cuda(),
        gaussian_covariances=covariances,
        gaussian_sh_coefficients=color.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, N, 3).unsqueeze(-1).cuda(),
        gaussian_opacities=opacities.unsqueeze(0).expand(B*T, -1).cuda(),
        scale_invariant=False,
        use_sh=False
    )
    color = color.reshape(B, T, 3, color.shape[-2], color.shape[-1])

    return color

def gs_render_batch(
        intrinsic, 
        extrinsic,  
        image_shape, 
        means, 
        scale,
        rotation,
        color, 
        opacities
    ):
    B, T, N, _ = means.shape
    background_color = torch.Tensor([0.0, 0.0, 0.0]).cuda()
    covariances = build_covariance(
        scale.unsqueeze(0).cuda(), 
        rotation.unsqueeze(0).cuda()
    ).unsqueeze(1).expand(B*T, N, -1, -1)
    
    color = render_cuda(
        extrinsics=extrinsic.unsqueeze(0).expand(B*T, -1, -1).cuda(),
        intrinsics=intrinsic.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, 3, 3).cuda(),
        near=torch.Tensor([0.5]).expand(B*T).cuda(),
        far=torch.Tensor([1000.0]).expand(B*T).cuda(),
        image_shape=image_shape,
        background_color=background_color.unsqueeze(0).expand(B*T, -1),
        gaussian_means=means.reshape(B*T, N, 3).cuda(),
        gaussian_covariances=covariances,
        gaussian_sh_coefficients=color.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, N, 3).unsqueeze(-1).cuda(),
        gaussian_opacities=opacities.unsqueeze(0).expand(B*T, -1).cuda(),
        scale_invariant=False,
        use_sh=False
    )
    color = color.reshape(B, T, 3, color.shape[-2], color.shape[-1])

    return color


def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)
