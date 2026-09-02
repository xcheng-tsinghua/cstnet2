import math
import unittest

import torch

from functional.constraints import _fit_cone, _fit_cylinder, estimate_normals_pca


def _surface_grid(theta_start, theta_stop, axial_start, axial_stop):
    theta = torch.linspace(theta_start, theta_stop, 25, dtype=torch.float64)
    axial = torch.linspace(axial_start, axial_stop, 17, dtype=torch.float64)
    theta_grid, axial_grid = torch.meshgrid(theta, axial, indexing="ij")
    return theta_grid.reshape(-1), axial_grid.reshape(-1)


def _basis(axis):
    helper = torch.zeros_like(axis)
    helper[int(axis.abs().argmin())] = 1.0
    first = torch.nn.functional.normalize(torch.cross(axis, helper, dim=0), dim=0)
    second = torch.nn.functional.normalize(torch.cross(axis, first, dim=0), dim=0)
    return first, second


class ConstraintFittingTest(unittest.TestCase):
    def test_partial_cylinder_recovers_axis_radius_and_origin_foot(self):
        axis = torch.nn.functional.normalize(
            torch.tensor([0.31, -0.42, 0.85], dtype=torch.float64), dim=0
        )
        foot_seed = torch.tensor([0.27, -0.31, 0.13], dtype=torch.float64)
        foot = foot_seed - torch.dot(foot_seed, axis) * axis
        radius = 0.43
        theta, axial = _surface_grid(-0.75, 0.95, -0.8, 0.7)
        first, second = _basis(axis)
        radial = (
            torch.cos(theta).unsqueeze(1) * first
            + torch.sin(theta).unsqueeze(1) * second
        )
        points = foot + axial.unsqueeze(1) * axis + radius * radial

        fitted_axis, fitted_radius, fitted_foot = _fit_cylinder(points, radial)

        self.assertTrue(torch.allclose(fitted_axis, axis, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(fitted_radius), radius, places=6)
        self.assertTrue(torch.allclose(fitted_foot, foot, atol=1e-6, rtol=0.0))

        estimated_normals = estimate_normals_pca(points.unsqueeze(0), k=16)[0]
        estimated_axis, estimated_radius, estimated_foot = _fit_cylinder(
            points, estimated_normals
        )
        self.assertLess(float((estimated_axis - axis).norm()), 0.01)
        self.assertLess(abs(float(estimated_radius) - radius), 0.05)
        self.assertLess(float((estimated_foot - foot).norm()), 0.05)

    def test_truncated_partial_cone_recovers_unobserved_apex_and_angle(self):
        axis = torch.nn.functional.normalize(
            torch.tensor([0.23, 0.31, 0.92], dtype=torch.float64), dim=0
        )
        apex = torch.tensor([0.19, -0.24, -0.62], dtype=torch.float64)
        semi_angle = 0.36
        theta, axial = _surface_grid(-0.85, 1.05, 0.45, 1.35)
        first, second = _basis(axis)
        radial = (
            torch.cos(theta).unsqueeze(1) * first
            + torch.sin(theta).unsqueeze(1) * second
        )
        points = (
            apex
            + axial.unsqueeze(1) * axis
            + (axial * math.tan(semi_angle)).unsqueeze(1) * radial
        )
        normals = math.cos(semi_angle) * radial - math.sin(semi_angle) * axis
        # Local PCA normal signs are arbitrary. Exercise the spatial sign
        # propagation rather than handing the fitter pre-oriented normals.
        normals[::3] *= -1.0

        fitted_axis, fitted_angle, fitted_apex = _fit_cone(points, normals)

        self.assertTrue(torch.allclose(fitted_axis, axis, atol=1e-6, rtol=0.0))
        self.assertAlmostEqual(float(fitted_angle), semi_angle, places=6)
        self.assertTrue(torch.allclose(fitted_apex, apex, atol=1e-6, rtol=0.0))

        estimated_normals = estimate_normals_pca(points.unsqueeze(0), k=16)[0]
        estimated_axis, estimated_angle, estimated_apex = _fit_cone(
            points, estimated_normals
        )
        self.assertLess(float((estimated_axis - axis).norm()), 0.02)
        self.assertLess(abs(float(estimated_angle) - semi_angle), 0.02)
        self.assertLess(float((estimated_apex - apex).norm()), 0.06)


if __name__ == "__main__":
    unittest.main()
