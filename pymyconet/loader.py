from hardware import load_core_segment

def import_from_generic_v1(core_path: str, metadata, zip_file):
    layers = []

    for layer in metadata['layers']:
        layer_type, file = layer['type'], layer['file']

        loaded_core = load_core_segment(core_path + f".{file}", layer_type)

        try:
            new = loaded_core.load(zip_file, layer)
        except NotImplementedError:
            raise NotImplementedError(f"Failed to load '{core_path}.{file}' Core Loading not implemented")

        layers.append(new)

    return layers