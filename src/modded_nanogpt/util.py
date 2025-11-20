def next_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
