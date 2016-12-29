from struct import unpack


def section_parser(total_size, fmt_size, fmt):
    def func(buf):
        assert len(buf) % total_size == 0
        size = len(buf) // total_size
        for i in range(size):
            yield unpack(fmt, buf[i*total_size:i*total_size+fmt_size])
    return func


lump_parsers = {
    'LINEDEFS': section_parser(14, 10, 'HHHHH'),
    'VERTEXES': section_parser(4, 4, 'hh'),
}


def parse_wad_directory(buf):
    loc, size = unpack("ii", buf[:8])
    name = buf[8:].decode().rstrip('\x00')
    return name, size, loc


def parse_wad_buffer(buf):
    size, dir_ptr = unpack("ii", buf[4:12])
    for i in range(size):
        name, s, ptr = parse_wad_directory(buf[dir_ptr+16*i:dir_ptr+16*(i+1)])
        yield (name, buf[ptr:ptr+s])


def extract_maps(dit):
    res = {}
    current = None
    for name, content in dit:
        if name.startswith('MAP'):
            if current is not None:
                yield current, res
            res = {}
            current = name
        elif name in lump_parsers:
            parse = lump_parsers[name]
            res[name] = list(parse(content))
    if current is not None:
        yield current, res


def extract_map_lines(map_content):
    lines, verts = map_content['LINEDEFS'], map_content['VERTEXES']
    return [(verts[v1], verts[v2], r) for v1, v2, *r in lines]


def parse_all_maps(filename):
    wad = parse_wad_buffer(open(filename, 'rb').read())
    return {
        name: extract_map_lines(content)
        for name, content in extract_maps(wad)
    }


if __name__ == "__main__":
    from sys import argv
    print(parse_all_maps(argv[1]))
