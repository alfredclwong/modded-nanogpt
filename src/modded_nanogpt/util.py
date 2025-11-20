def next_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()
