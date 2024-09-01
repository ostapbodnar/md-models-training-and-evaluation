def has_intersection(range1_start, range1_end, range2_start, range2_end):
    """
    Check if two given index ranges have clear intersections.

    Parameters:
    - range1_start (int): Start index of the first range.
    - range1_end (int): End index of the first range.
    - range2_start (int): Start index of the second range.
    - range2_end (int): End index of the second range.

    Returns:
    - bool: True if there is a clear intersection, False if the ranges are subsets or equal.
    """
    # Check if there is a clear intersection
    if range1_start < range2_end and range2_start < range1_end:
        # Check if the ranges are not subsets or exactly equal
        if not (range1_start <= range2_start and range1_end >= range2_end) and \
                not (range2_start <= range1_start and range2_end >= range1_end):
            return True
    return False


if __name__ == '__main__':
    print(has_intersection(0, 5, 3, 7))  # Output: True
    print(has_intersection(0, 5, 5, 10))  # Output: False
    print(has_intersection(0, 10, 5, 10))  # Output: False
    print(has_intersection(0, 5, 0, 10))  # Output: False
    print(has_intersection(0, 5, 6, 10))  # Output: False
    print(has_intersection(0, 5, 0, 5))  # Output: False
