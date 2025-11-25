import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import DataFolder, OutputFileManager
from datatrove.pipeline.tokens import DocumentTokenizerMerger
from fsspec import AbstractFileSystem


@dataclass
class Dataset:
    path: str
    epochs: int = 1


class MergedDataFolder(DataFolder):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        fs: AbstractFileSystem | None = None,
        auto_mkdir: bool = True,
        **storage_options,
    ):
        named_paths = []
        for dataset in datasets:
            path = dataset.path
            name = Path(path).name
            epochs = dataset.epochs

            if epochs > 1:
                named_paths.extend((f"{name}_{index}", path) for index in range(epochs))
            else:
                named_paths.append((name, path))

        self._folders = {name: DataFolder(str(path), fs, auto_mkdir, **storage_options) for name, path in named_paths}
        self.auto_mkdir = auto_mkdir

    def folder_names(self) -> list[str]:
        return list(self._folders)

    def list_files(
        self,
        subdirectory: str = "",
        recursive: bool = True,
        glob_pattern: str | None = None,
        include_directories: bool = False,
    ) -> list[str]:
        return [
            # NULL byte separator since it is an illegal symbol in both Unix-like and Windows paths
            f"{name}\0{file}"
            for name, folder in self._folders.items()
            for file in folder.list_files(subdirectory, recursive, glob_pattern, include_directories)
        ]

    def resolve_paths(self, paths) -> list[str] | str:
        raise NotImplementedError()

    def get_output_file_manager(self, **kwargs) -> OutputFileManager:
        raise NotImplementedError()

    def _path_and_folder(self, path):
        prefix, path = path.split("\0", 1)
        return path, self._folders[prefix]

    def open(self, path, mode="rb", *args, **kwargs):
        path, folder = self._path_and_folder(path)
        return folder.open(path, mode, *args, **kwargs)

    def is_local(self):
        return all(folder.is_local() for folder in self._folders.values())

    def isfile(self, path):
        path, folder = self._path_and_folder(path)
        return folder.isfile(path)

    def __repr__(self):
        return f"<{self.__class__.__qualname__} tracking {len(self._folders)} folders>"


def nanotron_merge(nanotron_data_dirs: list[Dataset], output_path: str) -> None:
    data_folder = MergedDataFolder(nanotron_data_dirs)
    merging_executor = LocalPipelineExecutor([DocumentTokenizerMerger(data_folder, output_path, "merged", seed=42)])
    merging_executor.run()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "source_datasets",
        help="JSON file containing paths to nanosets along with the desired number of epochs which should be merged and shuffled",
    )
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    return args


def main(args):
    with open(args.source_datasets, "r", encoding="utf-8") as file:
        datasets = json.load(file)

    nanotron_merge([Dataset(**dataset) for dataset in datasets], args.output)


if __name__ == "__main__":
    _args = get_args()
    main(_args)
