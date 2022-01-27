""""
Functions to enhance the terminal output readability
"""


def print_rd(print_input, print_style: str = "both"):
    """ "
    print_style = ["both", "above", "below"]
    """
    if print_style == "both":
        print("-----")
        print(print_input)
        print("-----")
    elif print_style == "above":
        print("-----")
        print(print_input)
    elif print_style == "below":
        print(print_input)
        print("-----")
    else:
        raise TypeError(
            f'Expected print_style = ["both", "above", "below"]. Got {print_style} instead.'
        )


def input_rd(user_input):
    print("-----")
    user_output = input(user_input)
    return user_output
