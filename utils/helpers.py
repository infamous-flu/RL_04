from typing import Dict, List, Optional, Tuple


def nice_box(width: int, contents: Optional[List[Tuple[str, str]]] = None, margin: int = 2,
             thick: bool = False, upper: bool = True, lower: bool = True, sides: bool = True) -> str:
    """
    Creates a nice decorative text box with the following options:

    Args:
        width (int): The total width of the box (including margins).
        contents (list, optional): A list of strings to be displayed inside the box. 
            Each string can optionally have a second element specifying alignment 
            ('c' for center, 'r' for right, default is left). Defaults to None.
        margin (int, optional): The margin size around the contents. Defaults to 2.
        thick (bool, optional): Boolean, determines if thick or thin box characters are used. Defaults to False.
        upper (bool, optional): Boolean, controls if the top border is drawn. Defaults to True.
        lower (bool, optional): Boolean, controls if the bottom border is drawn. Defaults to True.
        sides (bool, optional): Boolean, controls if the side borders are drawn. Defaults to True.

    Returns:
        str: The complete box string.
    """
    if not isinstance(width, int) or not isinstance(margin, int):
        raise TypeError('Width and margin must be integers.')

    if width <= 2 * margin:
        raise ValueError('Width must be greater than twice the margin.')

    contents = [] if contents is None else contents

    if not isinstance(contents, list):
        raise TypeError('Contents must be a list of tuples.')

    # Dictionary to map between thick/thin box characters
    box_char: Dict[str, str] = {'H': '═', 'V': '║', 'TL': '╔', 'TR': '╗',
                                'BL': '╚', 'BR': '╝'} if thick \
        else {'H': '─', 'V': '│', 'TL': '┌', 'TR': '┐',
              'BL': '└', 'BR': '┘'}

    # Calculate inner width for content display area
    inner_width: int = width-margin*2

    box_str: str = ''

    if upper:
        # Add top border
        box_str += (box_char['TL'] if sides else ' ') + \
            box_char['H'] * width + (box_char['TR'] if sides else ' ')

    for content in contents:
        if not isinstance(content, tuple) or len(content) != 2 or \
                not isinstance(content[0], str) or not isinstance(content[1], str):
            raise ValueError(
                'Contents must be a list of tuples with (string, alignment) format.')
        text: str = content[0]
        alignment: str = content[1]

        # Truncate content if it's longer than the inner width
        text = text[:inner_width-3] + \
            '...' if len(text) > (inner_width) else text

        # Calculate spaces for centering
        spaces = inner_width - len(text)
        halfsp = spaces // 2
        match alignment:
            case 'c':
                text = ' ' * halfsp + text + ' ' * (spaces - halfsp)
            case 'r':
                text = ' ' * spaces + text
            case _:
                text = text + ' ' * spaces

        # Build the content string with margins and vertical bar
        box_str += '\n' + (box_char['V'] if sides else ' ') + ' ' * \
            margin + text + ' ' * margin + (box_char['V'] if sides else ' ')

    if lower:
        # Add bottom border
        box_str += '\n' + (box_char['BL'] if sides else ' ') + \
            box_char['H'] * width + (box_char['BR'] if sides else ' ')

    return box_str
