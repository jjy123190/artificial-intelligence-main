"""Read the verified original mini-ImageNet tar files in order without filesystem seeks."""
import io
import json
import tarfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from .data import ImageSplit, digest_file, selection_digest


def load_archive(root, artifacts, name):
    manifest = json_from(artifacts / "dataset_manifest.json")["splits"][name]
    archive = root / "archives" / f"{name}.tar"
    if digest_file(archive) != manifest["archive"]["sha256"]:
        raise ValueError(f"Archive checksum mismatch for {name}")
    names = sorted(manifest["class_counts"])
    images = np.empty((manifest["images"], 84, 84, 3), dtype=np.uint8)
    ids, labels = [], []
    with tarfile.open(archive, "r|") as stream:
        for member in stream:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            split, cls, _ = path.parts
            if not member.isfile() or split != name or cls not in names:
                raise ValueError(f"Unexpected mini-ImageNet record {path}")
            with Image.open(io.BytesIO(stream.extractfile(member).read())) as image:
                if image.size != (84, 84):
                    raise ValueError(f"Unexpected resolution for {path}")
                images[len(ids)] = np.asarray(image.convert("RGB"))
            ids.append(path.as_posix())
            labels.append(names.index(cls))
    if len(ids) != len(images) or len(set(ids)) != len(ids):
        raise ValueError("Incomplete or duplicated records")
    result = ImageSplit(images, np.asarray(labels, dtype=np.int64), ids, names, name)
    ordered = ImageSplit(None, None, sorted(ids), names, name)
    protocol = json_from(artifacts / "protocol.json")
    expected = (json_from(artifacts / "continued_validation.json")["validation_ids_sha256"]
                if name == "val" else protocol[name + "_selection_sha256"])
    if selection_digest(ordered) != expected:
        raise ValueError("Image IDs differ from the released protocol")
    return result


def json_from(path):
    return json.loads(path.read_bytes())
