"""Модуль алгоритма поиска подстрок в строке (Бойера-Мура-Хорспула).

Реализует алгоритм Бойера-Мура-Хорспула (BMH),
декоратор для логирования времени выполнения и функцию search.
"""

import functools
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Декоратор для логирования времени выполнения функции."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Функция %s выполнена за %.6f секунд (%.3f мс)",
            func.__name__,
            elapsed_time,
            elapsed_time * 1000,
        )
        return result

    return wrapper


def deduplicate_patterns(
    patterns: Sequence[str],
    case_sensitivity: bool = False,
) -> List[str]:
    """Удаляет повторяющиеся подстроки, сохраняя исходный порядок.

    При поиске без учета регистра подстроки, отличающиеся только регистром,
    также считаются повторами.
    """
    seen: Set[str] = set()
    unique_patterns: List[str] = []
    for pattern in patterns:
        key = pattern if case_sensitivity else pattern.lower()
        if key not in seen:
            seen.add(key)
            unique_patterns.append(pattern)
    return unique_patterns


def compute_shift_table(pattern: str) -> Dict[str, int]:
    """Вычисление таблицы сдвигов для алгоритма Бойера-Мура-Хорспула."""
    pattern_len = len(pattern)
    shift_table: Dict[str, int] = {}
    for idx in range(pattern_len - 1):
        shift_table[pattern[idx]] = pattern_len - 1 - idx
    return shift_table


def bmh_search(text: str, pattern: str) -> List[int]:
    """Поиск всех вхождений подстроки алгоритмом Бойера-Мура-Хорспула."""
    text_len, pattern_len = len(text), len(pattern)
    if not pattern or not text or pattern_len > text_len:
        return []

    shift_table = compute_shift_table(pattern)
    matches: List[int] = []
    text_idx = 0

    while text_idx <= text_len - pattern_len:
        pat_idx = pattern_len - 1
        while pat_idx >= 0 and pattern[pat_idx] == text[text_idx + pat_idx]:
            pat_idx -= 1
        if pat_idx < 0:
            matches.append(text_idx)
            if pattern_len > 1:
                text_idx += shift_table.get(
                    text[text_idx + pattern_len - 1], pattern_len
                )
            else:
                text_idx += 1
        else:
            text_idx += shift_table.get(
                text[text_idx + pattern_len - 1], pattern_len
            )

    return matches


def _collect_matches(
    text: str,
    patterns: Sequence[str],
    search_patterns: Sequence[str],
) -> List[Tuple[int, int, str]]:
    """Сбор всех вхождений паттернов алгоритмом Бойера-Мура-Хорспула."""
    matches: List[Tuple[int, int, str]] = []
    for pat_idx, (orig_pat, pat) in enumerate(zip(patterns, search_patterns)):
        if not pat:
            continue
        for pos in bmh_search(text, pat):
            matches.append((pos, pat_idx, orig_pat))
    return matches


@log_execution_time
def search(
    string: str,
    sub_string: Union[str, List[str], Tuple[str, ...]],
    case_sensitivity: bool = False,
    method: str = "first",
    count: Optional[int] = None,
) -> Optional[Union[Tuple[int, ...], Dict[str, Optional[Tuple[int, ...]]]]]:
    """Поиск всех вхождений подстроки или набора подстрок в строке алгоритмом Бойера-Мура-Хорспула.

    Параметры:
        string: Исходная строка, в которой производится поиск.
        sub_string: Одна подстрока (str) или последовательность подстрок.
        case_sensitivity: Флаг чувствительности к регистру (по умолчанию False).
        method: Направление поиска: 'first' (с начала) или 'last' (с конца).
        count: Ограничение k на число находимых вхождений.

    Возвращает:
        None, если строка пуста или совпадений нет (для одиночной подстроки);
        Кортеж индексов для одиночной подстроки;
        Словарь вида {подстрока: кортеж_индексов_или_None} для набора подстрок.
    """
    if not string:
        return None

    if method not in ("first", "last"):
        raise ValueError(
            f"Неизвестное значение method: '{method}'. "
            "Ожидается 'first' или 'last'"
        )

    is_single = isinstance(sub_string, str)
    patterns = (
        [sub_string]
        if is_single
        else deduplicate_patterns(sub_string, case_sensitivity)
    )

    search_text = string if case_sensitivity else string.lower()
    search_patterns = [p if case_sensitivity else p.lower() for p in patterns]

    all_matches = _collect_matches(search_text, patterns, search_patterns)

    if method == "first":
        all_matches.sort(key=lambda item: (item[0], item[1]))
    else:
        all_matches.sort(key=lambda item: (-item[0], item[1]))

    if count is not None:
        if count <= 0:
            all_matches = []
        else:
            all_matches = all_matches[:count]

    if is_single:
        if not all_matches:
            return None
        return tuple(match[0] for match in all_matches)

    grouped_results: Dict[str, List[int]] = {pat: [] for pat in patterns}
    for match in all_matches:
        grouped_results[match[2]].append(match[0])

    return {
        pat: tuple(grouped_results[pat]) if grouped_results[pat] else None
        for pat in patterns
    }
