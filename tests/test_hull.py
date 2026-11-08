import numpy as np
from src.hull_checks import point_in_convex_hull, any_majority_inside

def test_point_inside_square():
    square = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    inside_pt = np.array([0.5, 0.5])
    outside_pt = np.array([1.5, 0.5])
    
    assert point_in_convex_hull(inside_pt, square) == True
    assert point_in_convex_hull(outside_pt, square) == False

def test_any_majority_inside():
    hull = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
    majority_pts = np.array([[5, 5], [1, 1], [-1, -1]])
    
    assert any_majority_inside(hull, majority_pts) == True