import subprocess

from scripts.deploy_space import committed_changes


def test_space_upload_uses_only_committed_blobs(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args]).decode().strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    (tmp_path / 'app.txt').write_bytes(b'original\n')
    git('add', '.')
    git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD')
    (tmp_path / 'app.txt').write_bytes(b'updated\n')
    (tmp_path / 'paper.pdf').write_bytes(b'%PDF-1.7\x00\x01')
    git('add', '.')
    git('commit', '-qm', 'release')
    (tmp_path / 'app.txt').write_bytes(b'UNCOMMITTED\n')
    (tmp_path / 'secret.env').write_text('must not upload')
    additions, deletions = committed_changes(tmp_path, base, 'HEAD')
    assert additions['app.txt'] == b'updated\n'
    assert additions['paper.pdf'] == b'%PDF-1.7\x00\x01'
    assert 'secret.env' not in additions
    assert deletions == []
