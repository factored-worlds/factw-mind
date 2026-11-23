def word(value: object) -> str:
    if type(value) is not str or len(value) != 5 or not all("A" <= c <= "Z" for c in value):
        raise ValueError("Words must contain five uppercase ASCII letters")
    return value


def vocabulary(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Expected a nonempty vocabulary")
    words = tuple(word(value) for value in values)
    if len(set(words)) != len(words):
        raise ValueError("Duplicate vocabulary entry")
    return tuple(sorted(words))


def feedback(guess: str, candidate: str) -> tuple[int, ...]:
    word(guess)
    word(candidate)
    codes = [0] * 5
    remaining = list(candidate)
    for i, letter in enumerate(guess):
        if candidate[i] == letter:
            codes[i] = 2
            remaining[i] = None
    for i, letter in enumerate(guess):
        if codes[i] == 0 and letter in remaining:
            codes[i] = 1
            remaining[remaining.index(letter)] = None
    return tuple(codes)
