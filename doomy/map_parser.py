import json


def wad_to_walls(string):
    """
    Returns a list of walls.
    Every wall is a pair of points (both ends of the wall)
    Every point is a pair (x, y)
    Tadaaa!
    """
    lines = string.split("\n")
    lines = [l if l.find("//") == -1 else l[:l.find("//")] for l in lines]
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and "namespace" not in l]

    sections = []

    i = 0
    while i < len(lines):
        name = lines[i]
        content = {}
        i += 2
        while lines[i] != "}":
            line = lines[i]
            key, value = line.split("=", 1)

            content[key.strip()] = json.loads(value.strip().rstrip(";"))
            i += 1

        sections.append(
            (name, content)
        )
        i += 1

    vertices = []
    blocking_lines = []

    for name, content in sections:
        if name == 'vertex':
            vertices.append(
                (content['x'], content['y'])
            )

    for name, content in sections:
        if name == 'linedef':
            if content.get('blocking', False):
                v1 = vertices[content['v1']]
                v2 = vertices[content['v2']]
                blocking_lines.append(
                    (v1, v2)
                )

    return blocking_lines


def parse(filename):
    with open(filename) as fp:
        return wad_to_walls(fp.read())
