"""Tests for geometry utilities (point_in_polygon, bbox_aspect)."""

import os
import sys

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

from core.geometry import bbox_aspect_h_over_w, point_in_polygon


def test_point_inside_square():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon(50, 50, polygon) is True


def test_point_outside_square():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon(150, 150, polygon) is False


def test_point_on_edge():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon(50, 0, polygon) is True


def test_point_at_vertex():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon(0, 0, polygon) is True


def test_polygon_triangle():
    polygon = [[0, 0], [100, 0], [50, 100]]
    assert point_in_polygon(50, 50, polygon) is True
    assert point_in_polygon(0, 50, polygon) is False


def test_less_than_3_vertices():
    polygon = [[0, 0], [100, 0]]
    assert point_in_polygon(50, 50, polygon) is False


def test_bbox_aspect_normal():
    # Tall rectangle (standing person)
    assert bbox_aspect_h_over_w(0, 0, 50, 150) == 3.0


def test_bbox_aspect_wide():
    # Wide rectangle (fallen person)
    assert bbox_aspect_h_over_w(0, 0, 150, 50) == 1.0 / 3.0


def test_bbox_aspect_square():
    assert bbox_aspect_h_over_w(0, 0, 100, 100) == 1.0


def test_bbox_zero_width():
    assert bbox_aspect_h_over_w(0, 0, 0, 100) > 0  # Should not divide by zero


def test_concave_polygon():
    """Point inside a concave (L-notch) polygon."""
    polygon = [[0, 0], [100, 0], [100, 40], [40, 40], [40, 80], [0, 80]]
    assert point_in_polygon(20, 20, polygon) is True
    assert point_in_polygon(70, 60, polygon) is False  # Inside the notch (outside)
