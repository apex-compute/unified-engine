"""Audit the supported n/s inference subset against a local upstream Git clone.

No upstream Python is imported or executed. Requires PyYAML and a clone with
the baseline and pinned master commits. Training/export/AutoShape are outside
the native Andromeda runtime's scope.
"""
import argparse
import ast
import subprocess
from pathlib import Path

import yaml
from yolov5_common import YOLOV5_SOURCE_COMMIT, load_yolov5_config

BASELINE = "915bbf294bb74c859f0b41f1c23bc395014ea679"


class StripDocs(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def verify(repo):
    def read(commit, path):
        return subprocess.check_output(
            ['git', '-C', str(repo), 'show', f'{commit}:{path}'], text=True)
    for variant in ('n', 's'):
        load_yolov5_config(variant)
        path = f'models/yolov5{variant}.yaml'
        if yaml.safe_load(read(BASELINE, path)) != yaml.safe_load(read(YOLOV5_SOURCE_COMMIT, path)):
            raise RuntimeError(f'{path}: graph changed; update the native compiler')
        print(f'{path}: graph unchanged')
    for path, names in (
        ('models/common.py', ('Conv', 'Bottleneck', 'C3', 'SPPF', 'Concat')),
        ('models/yolo.py', ('Detect',)),
    ):
        def classes(commit):
            tree = ast.parse(read(commit, path))
            return {n.name: ast.dump(StripDocs().visit(n)) for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name in names}
        before, after = classes(BASELINE), classes(YOLOV5_SOURCE_COMMIT)
        for name in names:
            if name not in before or before[name] != after.get(name):
                raise RuntimeError(f'{path}:{name}: inference implementation changed')
            print(f'{path}:{name}: implementation unchanged (excluding documentation)')
    print(f'PASS: native inference subset compatible with master {YOLOV5_SOURCE_COMMIT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    verify(parser.parse_args().upstream)
