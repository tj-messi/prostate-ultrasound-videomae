import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from util.util import parse_config_json

ROOT = Path(__file__).resolve().parents[1]


def test_config_parser_keeps_urls_and_resume_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        'name': 'synthetic',
        'path': {'base_dir': str(tmp_path / 'experiments'), 'code': 'code',
                 'log_file': 'run.log', 'resume_state': None, 'trust_resume_state': False},
        'url': 'https://example.com/synthetic',
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    opt = parse_config_json(SimpleNamespace(local_rank=-1, config=str(path), phase='train'))
    assert opt['url'] == config['url']
    assert opt['path']['trust_resume_state'] is False
    assert Path(opt['path']['log_file']).is_file()


def test_all_python_sources_parse():
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


@pytest.mark.parametrize('script', ['test.py', 'train/run.py', 'pretrain/run_mae_pretraining.py'])
def test_cli_help(script):
    path = ROOT / script
    result = subprocess.run([sys.executable, str(path), '--help'], cwd=path.parent,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
