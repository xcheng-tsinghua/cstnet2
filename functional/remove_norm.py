"""
就地删除分类分割数据集中的法线属性

"就地删除点云 TXT 数据中索引为 8、9、10 的法线字段，"
"兼容分类数据和 MFCAD++ 分割数据。"

"""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


# 删除前的数据列数
CLASSIFICATION_OLD_COLUMNS = 15
SEGMENTATION_OLD_COLUMNS = 17

# 删除后的数据列数
CLASSIFICATION_NEW_COLUMNS = 12
SEGMENTATION_NEW_COLUMNS = 14

# 零基索引：nor_x、nor_y、nor_z
NORMAL_COLUMN_INDICES = {8, 9, 10}


@dataclass(frozen=True)
class ProcessResult:
    file_path: Path
    status: str
    point_count: int = 0
    message: str = ""


def remove_normal_columns(fields: list[str]) -> list[str]:
    """
    删除零基索引 8、9、10 的法线字段。

    删除前：
        0  1  2    xyz
        3          primitive type
        4  5  6    direction
        7          dimension
        8  9  10   normal
        11 12 13   location
        14         primitive index
        15         face index，可选
        16         segmentation label，可选

    删除后：
        0  1  2    xyz
        3          primitive type
        4  5  6    direction
        7          dimension
        8  9  10   location
        11         primitive index
        12         face index，可选
        13         segmentation label，可选
    """
    return [
        value
        for index, value in enumerate(fields)
        if index not in NORMAL_COLUMN_INDICES
    ]


def inspect_and_convert_lines(
    file_path: Path,
) -> tuple[list[str], str, int]:
    """
    读取并转换一个文件。

    Returns:
        converted_lines:
            转换后的全部文本行。
        status:
            converted / already_processed / empty。
        point_count:
            非空数据行数量。

    Raises:
        ValueError:
            文件包含不支持的列数或混合格式。
    """
    converted_lines: list[str] = []
    point_count = 0
    has_old_format = False
    has_new_format = False

    with file_path.open("r", encoding="utf-8") as input_file:
        for line_number, original_line in enumerate(input_file, start=1):
            stripped_line = original_line.strip()

            # 保留空行
            if not stripped_line:
                converted_lines.append("\n")
                continue

            fields = stripped_line.split()
            column_count = len(fields)
            point_count += 1

            if column_count in {
                CLASSIFICATION_OLD_COLUMNS,
                SEGMENTATION_OLD_COLUMNS,
            }:
                has_old_format = True
                fields = remove_normal_columns(fields)
                converted_lines.append(" ".join(fields) + "\n")

            elif column_count in {
                CLASSIFICATION_NEW_COLUMNS,
                SEGMENTATION_NEW_COLUMNS,
            }:
                has_new_format = True
                converted_lines.append(" ".join(fields) + "\n")

            else:
                raise ValueError(
                    f"第 {line_number} 行有 {column_count} 列，"
                    "预期为 15/17 列的旧格式，或 12/14 列的已处理格式。"
                )

    if point_count == 0:
        return converted_lines, "empty", 0

    # 同一个文件中不应同时出现删除前和删除后的格式。
    # 这样可以防止部分处理后的损坏文件被继续覆盖。
    if has_old_format and has_new_format:
        raise ValueError(
            "文件同时包含删除前和删除后的数据行，拒绝就地覆盖。"
        )

    if has_old_format:
        return converted_lines, "converted", point_count

    return converted_lines, "already_processed", point_count


def write_atomic(
    file_path: Path,
    lines: list[str],
    backup_suffix: str | None,
) -> None:
    """
    在原文件所在目录写入临时文件，再原子替换原文件。

    backup_suffix 非空时，会先创建备份，例如：
        sample.txt -> sample.txt.bak
    """
    if backup_suffix:
        backup_path = file_path.with_name(file_path.name + backup_suffix)
        shutil.copy2(file_path, backup_path)

    temp_path: Path | None = None

    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            dir=file_path.parent,
            text=True,
        )
        temp_path = Path(temp_name)

        with os.fdopen(
            file_descriptor,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            output_file.writelines(lines)
            output_file.flush()
            os.fsync(output_file.fileno())

        # 保留原文件权限
        shutil.copymode(file_path, temp_path)

        # 同一文件系统内原子替换
        os.replace(temp_path, file_path)
        temp_path = None

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def process_file(
    file_path: Path,
    dry_run: bool,
    backup_suffix: str | None,
) -> ProcessResult:
    try:
        converted_lines, status, point_count = inspect_and_convert_lines(
            file_path
        )

        if status == "converted" and not dry_run:
            write_atomic(
                file_path=file_path,
                lines=converted_lines,
                backup_suffix=backup_suffix,
            )

        return ProcessResult(
            file_path=file_path,
            status=status,
            point_count=point_count,
        )

    except Exception as exception:
        return ProcessResult(
            file_path=file_path,
            status="error",
            message=str(exception),
        )


def collect_txt_files(
    root_path: Path,
    recursive: bool,
) -> list[Path]:
    if root_path.is_file():
        if root_path.suffix.lower() != ".txt":
            raise ValueError(f"输入文件不是 TXT 文件：{root_path}")
        return [root_path]

    if not root_path.is_dir():
        raise FileNotFoundError(f"路径不存在：{root_path}")

    iterator = root_path.rglob("*.txt") if recursive else root_path.glob("*.txt")
    return sorted(path for path in iterator if path.is_file())


def remove_norm_(
        root_path: Path,
        no_recursive: bool = False,
        workers: int = 0,
        dry_run: bool = False,
        backup: bool = False,
        ) -> int:
    """
    就地删除点云 TXT 文件中索引为 8、9、10 的法线字段。

    支持：
        分类数据：15 列 -> 12 列
        分割数据：17 列 -> 14 列

    Args:
        root_path:
            数据集根目录或单个 TXT 文件。
        no_recursive:
            是否仅处理根目录下的 TXT 文件。
        workers:
            并行线程数。小于等于 0 时使用 CPU 逻辑核心数减 1。
        dry_run:
            是否仅检查文件而不实际修改。
        backup:
            是否在修改前创建 .bak 备份。

    Returns:
        0:
            全部处理成功。
        1:
            存在错误文件或输入路径无效。
    """
    root_path = Path(root_path).expanduser().resolve()

    recursive = not no_recursive
    backup_suffix = ".bak" if backup else None

    cpu_count = os.cpu_count() or 1
    worker_count = (
        workers
        if workers > 0
        else max(1, cpu_count - 1)
    )

    try:
        txt_files = collect_txt_files(
            root_path=root_path,
            recursive=recursive,
        )
    except Exception as exception:
        print(f"[ERROR] {exception}")
        return 1

    if not txt_files:
        print(f"未找到 TXT 文件：{root_path}")
        return 0

    print(f"数据集路径：{root_path}")
    print(f"找到文件数：{len(txt_files)}")
    print(f"线程数：{worker_count}")
    print(f"模式：{'试运行，不修改文件' if dry_run else '就地修改'}")
    print(f"备份：{'启用' if backup else '禁用'}")

    converted_count = 0
    already_processed_count = 0
    empty_count = 0
    error_count = 0
    total_point_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_file,
                file_path,
                dry_run,
                backup_suffix,
            )
            for file_path in txt_files
        ]

        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="删除法线属性",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.2,
        ) as progress_bar:
            for future in progress_bar:
                result = future.result()

                if result.status == "converted":
                    converted_count += 1
                    total_point_count += result.point_count

                elif result.status == "already_processed":
                    already_processed_count += 1

                elif result.status == "empty":
                    empty_count += 1

                else:
                    error_count += 1
                    tqdm.write(
                        f"[ERROR] {result.file_path}: "
                        f"{result.message}"
                    )

                progress_bar.set_postfix(
                    {
                        "converted": converted_count,
                        "skipped": already_processed_count,
                        "empty": empty_count,
                        "errors": error_count,
                        "points": total_point_count,
                    },
                    refresh=False,
                )

    print()
    print("处理完成：")
    print(
        f"  {'待处理' if dry_run else '已处理'}文件："
        f"{converted_count}"
    )
    print(f"  已是新格式文件：{already_processed_count}")
    print(f"  空文件：{empty_count}")
    print(f"  错误文件：{error_count}")
    print(f"  已转换点数：{total_point_count}")

    return 1 if error_count > 0 else 0


