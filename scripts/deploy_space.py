"""Deploy committed source changes through Hub APIs, including binary PDF assets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def committed_changes(root: Path, base: str, revision: str) -> tuple[dict[str, bytes], list[str]]:
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    # Resolve both inputs before diffing; working-tree and untracked files are
    # never upload inputs. NUL-delimited paths support spaces and Unicode.
    base = git('rev-parse', '--verify', base + '^{commit}').decode().strip()
    revision = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    added = git('diff', '--name-only', '-z', '--no-renames', '--diff-filter=ACMT', base, revision)
    removed = git('diff', '--name-only', '-z', '--no-renames', '--diff-filter=D', base, revision)
    additions = {name.decode('utf-8'): git('show', revision + ':' + name.decode('utf-8'))
                 for name in added.split(b'\0') if name}
    return additions, [name.decode('utf-8') for name in removed.split(b'\0') if name]


def hub_token() -> str:
    from huggingface_hub import get_token
    token = get_token()
    if token:
        return token
    result = subprocess.run(
        ['git', 'credential', 'fill'], input='protocol=https\nhost=huggingface.co\n\n',
        capture_output=True, text=True, timeout=20,
        env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never'),
    )
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not credential.get('password'):
        raise RuntimeError('No Hugging Face token or saved Git credential available')
    return credential['password']


def main():
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-base', required=True, help='Last source commit already deployed to this Space')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--space', default='NeerajCodz/aiBatteryLifeCycle')
    parser.add_argument('--resume', action='store_true', help='Resume a paused Space on its existing hardware')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    additions, deletions = committed_changes(root, args.source_base, args.revision)
    source = subprocess.check_output(['git', '-C', str(root), 'rev-parse', args.revision]).decode().strip()
    api = HfApi(token=hub_token())
    info = api.space_info(args.space)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)
                  for path, data in additions.items()]
    operations += [CommitOperationDelete(path_in_repo=path) for path in deletions]
    if operations:
        result = api.create_commit(
            repo_id=args.space, repo_type='space', parent_commit=info.sha,
            operations=operations, commit_message='Deploy Reviewer 4 application and research release',
            commit_description=f'Source commit: {source}\nSource base: {args.source_base}\nCommitted files only; binary research assets uploaded using Hub storage.',
        )
        print('Space commit:', result.oid, flush=True)
    if args.resume and api.get_space_runtime(args.space).stage == 'PAUSED':
        print('Resuming existing Space:', api.restart_space(args.space).stage, flush=True)
    print('Source commit:', source, flush=True)
    print('Runtime:', api.get_space_runtime(args.space).stage, flush=True)
    print('https://huggingface.co/spaces/' + args.space, flush=True)


if __name__ == '__main__':
    main()
