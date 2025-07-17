"""
Tests for functions and classes in ANNotate_tools.
"""

from panhumanpy.ANNotate_tools import if_full_consistent_hierarchy

##### test the concordance detection fn if_full_consistent_hierarchy under different conditions

def test_single_level_hierarchy():
    """Test with max_depth=1 (single node) - edge case"""
    cell_label = ['A']
    result = if_full_consistent_hierarchy(cell_label, 1)
    assert result, "Single level hierarchy should always be consistent"

def test_two_level_consistent():
    """Test basic two-level consistent hierarchy"""
    cell_label = ['A', 'A|B']
    result = if_full_consistent_hierarchy(cell_label, 2)
    assert result, "A -> A|B should be consistent"

def test_three_level_consistent():
    """Test basic three-level consistent hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C']
    result = if_full_consistent_hierarchy(cell_label, 3)
    assert result, "A -> A|B -> A|B|C should be consistent"

def test_deep_consistent_hierarchy():
    """Test deeper consistent hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C', 'A|B|C|D', 'A|B|C|D|E']
    result = if_full_consistent_hierarchy(cell_label, 5)
    assert result, "Deep consistent hierarchy should return True"

def test_shared_node_path_jump_bug():
    """Test an important bug case - jumping between different paths to shared node"""
    cell_label = ['A', 'A|B', 'A|B|X', 'A|C|X|Y']
    result = if_full_consistent_hierarchy(cell_label, 4)
    assert not result, "Jumping from A|B|X to A|C|X|Y should be inconsistent"

def test_inconsistent_at_level_one():
    """Test inconsistency at first level transition"""
    cell_label = ['A', 'B|C']
    result = if_full_consistent_hierarchy(cell_label, 2)
    assert not result, "A -> B|C should be inconsistent"

def test_inconsistent_at_level_two():
    """Test inconsistency at second level transition"""
    cell_label = ['A', 'A|B', 'A|C|D']
    result = if_full_consistent_hierarchy(cell_label, 3)
    assert not result, "A|B -> A|C|D should be inconsistent"

def test_inconsistent_deep_hierarchy():
    """Test inconsistency deep in hierarchy"""
    cell_label = ['A', 'A|B', 'A|B|C', 'A|B|X|D']
    result = if_full_consistent_hierarchy(cell_label, 4)
    assert not result, "A|B|C -> A|B|X|D should be inconsistent"