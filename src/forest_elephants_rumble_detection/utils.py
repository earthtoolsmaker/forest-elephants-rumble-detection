from pathlib import Path

import yaml


class MyDumper(yaml.Dumper):
    """Formatter for dumping yaml."""

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def yaml_read(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def yaml_write(to: Path, data: dict, dumper=MyDumper) -> None:
    """Writes a `data` dictionnary to the provided `to` path."""
    with open(to, "w") as f:
        yaml.dump(
            data=data,
            stream=f,
            Dumper=dumper,
            default_flow_style=False,
            sort_keys=False,
        )
