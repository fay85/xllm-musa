"""Repair the known 1102-dev compiler wrapper inside an isolated build container.

Never run this on the host. Existing versions are checked before mutation.
"""
from pathlib import Path
import hashlib
import json


def main() -> None:
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Only run inside the task build container")
    path = Path("/usr/local/musa/tools/musamapping/mcc_wrapper")
    content = path.read_text()
    before = hashlib.sha256(content.encode()).hexdigest()
    original = "d681b06dfbd771d0ae015a837982524a4382ff75746ee565c56c3f14109fa7dd"
    fixed = "8d89fe992fb0363ab4e2f9f43f2d7d1cf289d6d38ec7ade8f86018ad2faf0257"
    if before == fixed:
        print(json.dumps({"status": "already_fixed", "sha256": fixed}))
        return
    if before != original:
        raise RuntimeError("Unknown wrapper: inspect its compiler exit-code handling first")
    changes = [
        ('    g++ "${grouped_args[@]}"\n    exit 0', '    exec g++ "${grouped_args[@]}"'),
        ('mcc "${filtered_args[@]}"\nexit 0', 'exec mcc "${filtered_args[@]}"'),
    ]
    for source, target in changes:
        if content.count(source) != 1:
            raise RuntimeError("Unexpected wrapper structure")
        content = content.replace(source, target, 1)
    if hashlib.sha256(content.encode()).hexdigest() != fixed:
        raise RuntimeError("Patched wrapper hash mismatch")
    backup = path.with_name(path.name + ".xiaopang1105.original")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(content)
    print(json.dumps({"status": "fixed", "original_sha256": before, "sha256": fixed}))


if __name__ == "__main__":
    main()
