
def debug(text, debug_mode=True):
    """
    text: the text that preceed the debug message
    var: list of variables whose values are to be displayed
    """
    if debug_mode:
        print(text)


def from_molecules_to_uM(molecules, molecules_per_uM):
    return molecules / molecules_per_uM

def from_uM_to_molecules(uM, molecules_per_uM):
    return uM * molecules_per_uM
