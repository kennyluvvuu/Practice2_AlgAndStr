"""Консольная утилита для поиска подстрок в тексте или файле.

Использует алгоритм Бойера-Мура-Хорспула.

Поддерживает:
- Поиск одиночных и множественных подстрок
- Чувствительность/нечувствительность к регистру
- Поиск k первых вхождений (параметр count)
- Направление поиска: с начала (first) или с конца (last)
- Поиск в целевой строке или в файле
- Цветной вывод в терминал с уникальным цветом для каждой подстроки
- Ограничение вывода не более 10 строк с найденными совпадениями
- Замер и логирование времени выполнения
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

from search import search

# Палитра ANSI-цветов для выделения различных подстрок
COLOR_PALETTE: Tuple[str, ...] = (
    "\033[93m",  # Bright Yellow
    "\033[96m",  # Bright Cyan
    "\033[92m",  # Bright Green
    "\033[95m",  # Bright Magenta
    "\033[91m",  # Bright Red
    "\033[94m",  # Bright Blue
)
RESET_COLOR: str = "\033[0m"


def build_argument_parser() -> argparse.ArgumentParser:
    """Создает и настраивает парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Консольная утилита поиска подстрок в строке или файле "
            "(алгоритм Бойера-Мура-Хорспула)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-s",
        "--string",
        type=str,
        help="Целевая строка для поиска",
    )
    source_group.add_argument(
        "-f",
        "--file",
        type=str,
        help="Путь к текстовому файлу для поиска",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        nargs="+",
        required=True,
        help="Одна или несколько подстрок для поиска",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Включить чувствительность к регистру (по умолчанию нечувствительно)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["first", "last"],
        default="first",
        help="Направление поиска (first — с начала, last — с конца)",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=None,
        help="Максимальное количество совпадений k",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Отключить цветное выделение в консоли",
    )
    parser.add_argument(
        "--show-time",
        action="store_true",
        default=False,
        help="Выводить время выполнения поиска",
    )
    parser.add_argument(
        "-A",
        "--all",
        action="store_true",
        default=False,
        help="Показать все строки с совпадениями без ограничения в 10 строк",
    )
    return parser


def read_source_text(args: argparse.Namespace) -> str:
    """Считывает исходный текст из аргумента командной строки или файла."""
    if args.string is not None:
        return args.string
    try:
        with open(args.file, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except (OSError, UnicodeDecodeError) as err:
        print(f"Ошибка при чтении файла '{args.file}': {err}", file=sys.stderr)
        sys.exit(1)


def highlight_line(
    line: str,
    line_start_idx: int,
    matches: Sequence[Tuple[int, int, str]],
    use_color: bool,
) -> str:
    """Форматирует строку с цветным выделением вхождений подстрок."""
    if not use_color:
        return line

    line_end_idx = line_start_idx + len(line)
    line_segments: List[Tuple[int, int, str]] = []

    for start_pos, end_pos, color in matches:
        if start_pos < line_end_idx and end_pos > line_start_idx:
            local_start = max(0, start_pos - line_start_idx)
            local_end = min(len(line), end_pos - line_start_idx)
            line_segments.append((local_start, local_end, color))

    if not line_segments:
        return line

    line_segments.sort(key=lambda seg: (seg[0], seg[1]))

    result_chars: List[str] = []
    current_pos = 0

    for local_start, local_end, color in line_segments:
        local_start = max(local_start, current_pos)
        if local_start >= local_end:
            continue
        result_chars.append(line[current_pos:local_start])
        result_chars.append(color)
        result_chars.append(line[local_start:local_end])
        result_chars.append(RESET_COLOR)
        current_pos = local_end

    result_chars.append(line[current_pos:])
    return "".join(result_chars)


def extract_matched_intervals(
    patterns: List[str],
    raw_result: Optional[
        Union[Tuple[int, ...], Dict[str, Optional[Tuple[int, ...]]]]
    ],
) -> List[Tuple[int, int, str]]:
    """Формирует список интервалов (start, end, color) для подсветки."""
    matched_intervals: List[Tuple[int, int, str]] = []
    if raw_result is None:
        return matched_intervals

    if isinstance(raw_result, tuple):
        color = COLOR_PALETTE[0]
        pat_len = len(patterns[0])
        for pos in raw_result:
            matched_intervals.append((pos, pos + pat_len, color))
    elif isinstance(raw_result, dict):
        for idx, pat in enumerate(patterns):
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            pat_indices = raw_result.get(pat)
            if pat_indices:
                pat_len = len(pat)
                for pos in pat_indices:
                    matched_intervals.append((pos, pos + pat_len, color))

    return matched_intervals


def _find_matching_line_indices(
    lines: Sequence[str],
    line_offsets: Sequence[int],
    intervals: Sequence[Tuple[int, int, str]],
) -> List[int]:
    """Определяет индексы строк, в которых есть хотя бы одно совпадение."""
    matching: List[int] = []
    for line_idx, (line, start_idx) in enumerate(zip(lines, line_offsets)):
        end_idx = start_idx + len(line)
        if any(
            start < end_idx and end > start_idx
            for start, end, _ in intervals
        ):
            matching.append(line_idx)
    return matching


def print_colored_output(
    text: str,
    intervals: List[Tuple[int, int, str]],
    use_color: bool,
    max_lines: Optional[int] = 10,
) -> None:
    """Выводит строки с найденными совпадениями с опциональным лимитом max_lines."""
    if not intervals:
        return

    lines = text.splitlines()
    line_offset = 0
    line_offsets: List[int] = []

    for line in lines:
        line_offsets.append(line_offset)
        line_offset += len(line) + 1

    matching_indices = _find_matching_line_indices(
        lines, line_offsets, intervals
    )

    print("\nНайденные фрагменты текста:")
    displayed_indices = (
        matching_indices[:max_lines]
        if max_lines is not None
        else matching_indices
    )
    for line_idx in displayed_indices:
        highlighted = highlight_line(
            lines[line_idx], line_offsets[line_idx], intervals, use_color
        )
        print(f"[{line_idx + 1:3d}] {highlighted}")

    if max_lines is not None and len(matching_indices) > max_lines:
        print(
            f"... (показано первые {max_lines} строк с совпадениями "
            f"из {len(matching_indices)})"
        )


def main() -> None:
    """Основная функция запуска консольной утилиты."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.show_time:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    source_text = read_source_text(args)
    pattern_arg: Union[str, List[str]] = (
        args.pattern[0] if len(args.pattern) == 1 else args.pattern
    )

    search_result = search(
        string=source_text,
        sub_string=pattern_arg,
        case_sensitivity=args.case_sensitive,
        method=args.method,
        count=args.count,
    )

    print(f"Результат поиска: {search_result}")

    matched_intervals = extract_matched_intervals(args.pattern, search_result)
    print_colored_output(
        text=source_text,
        intervals=matched_intervals,
        use_color=not args.no_color,
        max_lines=None if args.all else 10,
    )


if __name__ == "__main__":
    main()
