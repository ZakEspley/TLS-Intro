import numpy as np
def prefix(x, digits=None):
    prefixes = {
        24: "Y",
        21: "Z",
        18: "E",
        15: "P",
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: "µ",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    }
    exponent = np.floor(np.log10(x))
    roundedExp = int(exponent/3)*3
    if roundedExp not in prefixes:
        raise ValueError(f"The value: {x} is too large or small for a prefix.")
    prefix = prefixes[roundedExp]
    value = x/(10**roundedExp)
    if digits is None:
        return f"{value} {prefix}"
    else:
        return f"{value:0.{digits}f} {prefix}"

