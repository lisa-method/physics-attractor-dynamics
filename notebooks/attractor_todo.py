import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Практическая тетрадь: Physics Attractor

    Эта Marimo-тетрадь предназначена только для анализа и экспериментов.

    Теория находится отдельно: [`docs/THEORY.md`](../docs/THEORY.md).

    **Принцип работы:** выполняем один TODO, проверяем его результат и только
    после этого переходим к следующему. SINDy будем использовать через PySINDy,
    а не реализовывать вручную.
    """)
    return


@app.cell
def _():
    # Импорты для всех запланированных этапов проекта.
    from pathlib import Path
    from zipfile import ZipFile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pysindy as ps
    from scipy.integrate import solve_ivp
    from scipy.signal import savgol_filter
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupShuffleSplit

    return Path, np, pd, plt


@app.cell
def _(Path, pd):
    # Архив скачан локально и исключён из Git через .gitignore.
    DATA_ARCHIVE = Path("data/physics-attractor-time-series.zip")

    if not DATA_ARCHIVE.exists():
        raise FileNotFoundError(
            "Не найден data/physics-attractor-time-series.zip. "
            "Сначала положи локальный архив датасета в папку data/."
        )

    raw_df = pd.read_csv(DATA_ARCHIVE, compression="zip")

    # Это исходные данные без преобразований. Их исходный порядок сохраняем,
    # чтобы не потерять порядок регистрации наблюдений до проверки времени.
    return DATA_ARCHIVE, raw_df


@app.cell(hide_code=True)
def _(DATA_ARCHIVE, mo, raw_df):
    dt = raw_df["time"].diff().dropna()
    empty_columns = raw_df.columns[raw_df.isna().all()].tolist()

    mo.md(
        f"""
        ## Датасет загружен

        - архив: `{DATA_ARCHIVE}`;
        - строк: **{len(raw_df):,}**;
        - исходные столбцы: `{', '.join(raw_df.columns)}`;
        - полностью пустые столбцы: **{len(empty_columns)}**;
        - время: от **{raw_df['time'].min():.3f}** до **{raw_df['time'].max():.3f}**;
        - шаг времени: от **{dt.min():.3f}** до **{dt.max():.3f}**.

        Данные показаны без очистки. Ниже удаляем только технически пустые
        столбцы, нормализуем имена и отдельно проверяем, допустима ли сортировка.
        """
    )
    return


@app.cell
def _(raw_df):
    # Первые пять исходных строк — без очистки и преобразований.
    raw_df.head()
    return


@app.cell
def _(raw_df):
    raw_df.isna().sum()
    return


@app.cell
def _(pd, raw_df):
    # Короткие x1/y1/x2/y2 совпадают с математическими обозначениями в теории
    # и станут понятными feature names в PySINDy. Названия диагностических
    # полей не уточняем, пока их физический смысл неизвестен.
    STATE_COLUMNS = ["x1", "y1", "x2", "y2"]
    DIAGNOSTIC_COLUMNS = ["distance", "angle1", "angle2"]

    _rename_columns = {
        "pos1x": "x1",
        "pos1y": "y1",
        "pos2x": "x2",
        "pos2y": "y2",
    }

    df = raw_df.dropna(axis="columns", how="all").copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_rename_columns)

    _expected_columns = ["time", *STATE_COLUMNS, *DIAGNOSTIC_COLUMNS]
    _missing_columns = sorted(set(_expected_columns) - set(df.columns))
    _unexpected_columns = sorted(set(df.columns) - set(_expected_columns))
    if _missing_columns or _unexpected_columns:
        raise ValueError(
            "Неожиданная схема данных. "
            f"Отсутствуют: {_missing_columns}; лишние: {_unexpected_columns}."
        )

    # Явное преобразование не подменяет ошибки пропусками: любой текст или
    # повреждённое число остановит ячейку с понятным исключением.
    df = df.loc[:, _expected_columns].apply(pd.to_numeric, errors="raise")
    df.index.name = "source_row"

    df.head()
    return DIAGNOSTIC_COLUMNS, STATE_COLUMNS, df


@app.cell
def _(pd):
    data_dictionary = pd.DataFrame(
        [
            ("time", "время наблюдения", "единица времени неизвестна", "модель/ось"),
            ("x1", "x-координата объекта 1", "единица координат неизвестна", "состояние"),
            ("y1", "y-координата объекта 1", "единица координат неизвестна", "состояние"),
            ("x2", "x-координата объекта 2", "единица координат неизвестна", "состояние"),
            ("y2", "y-координата объекта 2", "единица координат неизвестна", "состояние"),
            ("distance", "исходный диагностический признак", "физический смысл неизвестен", "не моделировать"),
            ("angle1", "исходный угловой признак 1", "физический смысл неизвестен", "не моделировать"),
            ("angle2", "исходный угловой признак 2", "физический смысл неизвестен", "не моделировать"),
        ],
        columns=["column", "working_description", "units_or_limitation", "role"],
    ).set_index("column")

    data_dictionary
    return (data_dictionary,)


@app.cell
def _(df, pd, raw_df):
    _time_diff = df["time"].diff()
    _empty_column_count = int(raw_df.isna().all(axis="rows").sum())
    _remaining_missing = int(df.isna().sum().sum())
    _duplicate_rows = int(df.duplicated().sum())
    _duplicate_times = int(df["time"].duplicated().sum())
    _time_order_issues = int(_time_diff.dropna().le(0).sum())
    _all_numeric = bool(df.dtypes.apply(pd.api.types.is_numeric_dtype).all())

    quality_checks = pd.DataFrame(
        [
            ("Удалены полностью пустые столбцы", _empty_column_count, _empty_column_count == 8),
            ("Пропуски после очистки", _remaining_missing, _remaining_missing == 0),
            ("Полные дубликаты строк", _duplicate_rows, _duplicate_rows == 0),
            ("Повторяющиеся значения time", _duplicate_times, _duplicate_times == 0),
            ("Невозрастающие шаги time (dt <= 0)", _time_order_issues, _time_order_issues == 0),
            ("Все содержательные столбцы числовые", _all_numeric, _all_numeric),
        ],
        columns=["check", "observed", "passed"],
    )

    time_step_summary = pd.Series(
        {
            "min": _time_diff.dropna().min(),
            "median": _time_diff.dropna().median(),
            "max": _time_diff.dropna().max(),
            "unique_values": _time_diff.dropna().nunique(),
        },
        name="dt",
    ).to_frame()
    return quality_checks, time_step_summary


@app.cell(hide_code=True)
def _(mo, quality_checks, time_step_summary):
    mo.vstack(
        [
            mo.md(r"""
            ## Результат очистки и решение по сортировке

            Координаты переименованы в `x1`, `y1`, `x2`, `y2`: эти имена
            соответствуют формулам в теории, компактны на графиках и напрямую
            подходят как имена признаков PySINDy. `time` оставлено явным, а
            `distance`, `angle1`, `angle2` не переименованы: их смысл пока не
            установлен, и новое содержательное имя создало бы ложную трактовку.

            **Глобальная сортировка не выполняется.** Исходный порядок строк —
            часть временных данных. В текущем файле `time` уже строго возрастает,
            поэтому сортировка ничего не исправит. Если в новом файле время
            сбросится (`dt <= 0`), это признак начала новой записи и будущая
            граница `segment_id`, а не повод перемешивать записи общей сортировкой.
            Сортировать по `time` можно только внутри уже подтверждённого сегмента,
            если отдельно установлено, что строки лишь записаны не по порядку и
            каждое значение времени уникально.
            """),
            mo.md("### Проверки качества"),
            quality_checks,
            mo.md("### Неравномерность временного шага"),
            time_step_summary,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Координаты как функции времени

    Серые траектории `x`–`y` выше — это только проекция движения на плоскость:
    там не видно, в какой момент пройдена каждая точка. На графиках ниже по
    горизонтали отложено настоящее `time`, поэтому можно проверить
    непрерывность координат напрямую.

    Слева — вся запись, справа — увеличенная окрестность одного из редких
    больших шагов времени. Красная вертикаль показывает точку после паузы
    записи.
    """)
    return


@app.cell
def _(df, plt):
    # Эти границы можно менять, чтобы рассмотреть любой фрагмент записи.
    _time_start = 808
    _time_end = 812
    _zoom_df = df[df["time"].between(_time_start, _time_end)]
    _zoom_candidates = _zoom_df[_zoom_df["time"].diff().gt(0.06)]

    _time_fig, _time_axes = plt.subplots(2, 2, figsize=(14, 7), sharey="row")

    for _row, (_x_column, _y_column, _object_name) in enumerate(
        [
            ("x1", "y1", "объект 1"),
            ("x2", "y2", "объект 2"),
        ]
    ):
        _overview_ax = _time_axes[_row, 0]
        _overview_ax.plot(df["time"], df[_x_column], linewidth=0.45, label="x")
        _overview_ax.plot(df["time"], df[_y_column], linewidth=0.45, label="y")
        _overview_ax.set_title(f"{_object_name}: вся запись")

        _zoom_ax = _time_axes[_row, 1]
        _zoom_ax.plot(_zoom_df["time"], _zoom_df[_x_column], label="x")
        _zoom_ax.plot(_zoom_df["time"], _zoom_df[_y_column], label="y")
        for _candidate_time in _zoom_candidates["time"]:
            _zoom_ax.axvline(
                _candidate_time,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label="точка после dt > 0.06",
            )
        _zoom_ax.set_title(f"{_object_name}: увеличение {_time_start}–{_time_end}")

        for _ax in (_overview_ax, _zoom_ax):
            _ax.set_xlabel("time")
            _ax.set_ylabel("координата")
            _ax.grid(alpha=0.25)
            _ax.legend()

    _time_fig.suptitle("Координаты объектов во времени")
    _time_fig.tight_layout()

    _time_fig
    return


@app.cell
def _(df, plt):
    _trajectory_fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, _x_column, _y_column, _title in [
        (_axes[0], "x1", "y1", "Траектория объекта 1"),
        (_axes[1], "x2", "y2", "Траектория объекта 2"),
    ]:
        _ax.plot(
            df[_x_column],
            df[_y_column],
            color="slategray",
            linewidth=0.5,
            alpha=0.7,
            label="траектория",
        )
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.set_title(_title)
        _ax.set_aspect("equal", adjustable="box")
        _ax.grid(alpha=0.25)
        _ax.legend()

    _trajectory_fig.suptitle("Движение в плоскости: время не является осью")
    _trajectory_fig.tight_layout()

    _trajectory_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Промежуточный вывод: проверка времени

    В исходном порядке строк время всегда возрастает: сбросов `time` нет.
    Четыре сравнительно больших временных шага проверены на графиках координат
    от времени и в плоскости `x`–`y`; заметных скачков положения не видно.

    Эта проверка исключает разрывы самого времени, но ещё не доказывает
    непрерывность траекторий. Дальше отдельно проверяем пространственный шаг между
    соседними положениями объектов.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пространственный шаг между соседними наблюдениями

    Для каждого объекта считаем евклидово расстояние между двумя соседними
    положениями. Общий диагностический признак `spatial_step` — максимум из шагов
    двух объектов: так разрыв будет заметен, даже если резко переместился только
    один объект.

    На этом этапе только рассчитываем признаки и смотрим на их распределение.
    Порог разрыва и `segment_id` выберем отдельно после проверки графика и таблицы.
    """)
    return


@app.cell
def _(df, np):
    step_df = df.copy()
    step_df["dt"] = step_df["time"].diff()

    step_df["step_object1"] = np.hypot(
        step_df["x1"].diff(),
        step_df["y1"].diff(),
    )
    step_df["step_object2"] = np.hypot(
        step_df["x2"].diff(),
        step_df["y2"].diff(),
    )
    step_df["spatial_step"] = step_df[
        ["step_object1", "step_object2"]
    ].max(axis="columns")

    largest_spatial_steps = step_df.nlargest(15, "spatial_step")[
        ["time", "dt", "step_object1", "step_object2", "spatial_step"]
    ]
    return largest_spatial_steps, step_df


@app.cell
def _(largest_spatial_steps):
    # Крупнейшие шаги: здесь удобно искать разрыв между обычным движением
    # и редкими пространственными скачками.
    largest_spatial_steps
    return


@app.cell
def _(plt, step_df):
    _step_fig, _step_ax = plt.subplots(figsize=(14, 5))

    _step_ax.plot(
        step_df["time"],
        step_df["step_object1"],
        linewidth=0.55,
        alpha=0.75,
        label="объект 1",
    )
    _step_ax.plot(
        step_df["time"],
        step_df["step_object2"],
        linewidth=0.55,
        alpha=0.75,
        label="объект 2",
    )
    _step_ax.set_yscale("log")
    _step_ax.set_xlabel("time")
    _step_ax.set_ylabel("пространственный шаг")
    _step_ax.set_title("Шаг между соседними положениями объектов")
    _step_ax.grid(alpha=0.25)
    _step_ax.legend()
    _step_fig.tight_layout()

    _step_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Локальная проверка крупнейших шагов

    Выбери строку из списка, чтобы рассмотреть небольшой фрагмент движения вокруг
    неё. Верхние графики показывают координаты от времени. Средние показывают путь
    каждого объекта в плоскости `x`–`y`: синяя линия идёт до проверяемого шага,
    оранжевая — после, а красная стрелка показывает переход между двумя соседними
    точками, для которых рассчитан выбранный `spatial_step`.

    Пределы осей `x` и `y` зафиксированы по всему датасету и не меняются при
    переключении кандидата. Нижний график также использует один масштаб для всех
    кандидатов и показывает величину шага рядом с выбранной точкой.

    В списке оставлены 15 крупнейших шагов: так можно сравнить редкие большие
    скачки с верхней границей обычного движения, не выбирая порог заранее.
    """)
    return


@app.cell
def _(largest_spatial_steps, mo):
    spatial_candidate_labels = [
        (
            f"{_rank:02d}. строка {_index}: "
            f"time={_row['time']:.3f}, "
            f"spatial_step={_row['spatial_step']:.3f}"
        )
        for _rank, (_index, _row) in enumerate(
            largest_spatial_steps.iterrows(),
            start=1,
        )
    ]
    spatial_candidate = mo.ui.dropdown(
        options=spatial_candidate_labels,
        value=spatial_candidate_labels[0],
        label="Кандидат на разрыв",
    )
    spatial_candidate
    return spatial_candidate, spatial_candidate_labels


@app.cell
def _(
    largest_spatial_steps,
    spatial_candidate,
    spatial_candidate_labels,
    step_df,
):
    _candidate_rank = spatial_candidate_labels.index(spatial_candidate.value)
    selected_step_index = int(largest_spatial_steps.index[_candidate_rank])
    selected_step_row = step_df.loc[selected_step_index]

    _selected_position = step_df.index.get_loc(selected_step_index)
    _window_radius = 15
    _window_start = max(0, _selected_position - _window_radius)
    _window_stop = min(len(step_df), _selected_position + _window_radius + 1)
    local_step_df = step_df.iloc[_window_start:_window_stop]

    _neighbor_steps = local_step_df.loc[
        local_step_df.index != selected_step_index,
        "spatial_step",
    ].dropna()
    typical_neighbor_step = _neighbor_steps.median()
    selected_to_typical_ratio = (
        selected_step_row["spatial_step"] / typical_neighbor_step
    )
    return (
        local_step_df,
        selected_step_index,
        selected_step_row,
        selected_to_typical_ratio,
        typical_neighbor_step,
    )


@app.cell(hide_code=True)
def _(
    mo,
    selected_step_index,
    selected_step_row,
    selected_to_typical_ratio,
    typical_neighbor_step,
):
    mo.md(f"""
    **Проверяемый шаг:** строка после скачка — `{selected_step_index}`,
    `time = {selected_step_row['time']:.3f}`, `dt = {selected_step_row['dt']:.3f}`.

    - объект 1: `{selected_step_row['step_object1']:.3f}`;
    - объект 2: `{selected_step_row['step_object2']:.3f}`;
    - общий `spatial_step`: **`{selected_step_row['spatial_step']:.3f}`**.

    Медиана соседних шагов: `{typical_neighbor_step:.3f}`. Выбранный шаг больше
    неё в **{selected_to_typical_ratio:.1f} раза**.
    """)
    return


@app.cell
def _(local_step_df, plt, selected_step_index, selected_step_row, step_df):
    _candidate_time = selected_step_row["time"]
    _before_df = local_step_df[local_step_df.index < selected_step_index]
    _after_df = local_step_df[local_step_df.index >= selected_step_index]
    _previous_row = local_step_df.loc[selected_step_index - 1]

    _x_min = step_df[["x1", "x2"]].min().min()
    _x_max = step_df[["x1", "x2"]].max().max()
    _y_min = step_df[["y1", "y2"]].min().min()
    _y_max = step_df[["y1", "y2"]].max().max()
    _x_padding = 0.03 * (_x_max - _x_min)
    _y_padding = 0.03 * (_y_max - _y_min)

    _local_fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    _local_grid = _local_fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1.15, 0.75],
    )
    _time_axes = [
        _local_fig.add_subplot(_local_grid[0, 0]),
        _local_fig.add_subplot(_local_grid[0, 1]),
    ]
    _movement_axes = [
        _local_fig.add_subplot(_local_grid[1, 0]),
        _local_fig.add_subplot(_local_grid[1, 1]),
    ]
    _local_step_ax = _local_fig.add_subplot(_local_grid[2, :])

    for _ax, _x_column, _y_column, _title in [
        (_time_axes[0], "x1", "y1", "Объект 1: координаты от времени"),
        (_time_axes[1], "x2", "y2", "Объект 2: координаты от времени"),
    ]:
        _ax.plot(local_step_df["time"], local_step_df[_x_column], label="x")
        _ax.plot(local_step_df["time"], local_step_df[_y_column], label="y")
        _ax.axvline(
            _candidate_time,
            color="crimson",
            linestyle="--",
            linewidth=1.3,
            label="точка после шага",
        )
        _ax.set_xlabel("time")
        _ax.set_ylabel("координата")
        _ax.set_title(_title)
        _ax.grid(alpha=0.25)
        _ax.legend()

    for _ax, _x_column, _y_column, _step_column, _object_name in [
        (_movement_axes[0], "x1", "y1", "step_object1", "Объект 1"),
        (_movement_axes[1], "x2", "y2", "step_object2", "Объект 2"),
    ]:
        _ax.plot(
            _before_df[_x_column],
            _before_df[_y_column],
            marker="o",
            markersize=2.5,
            color="steelblue",
            label="до шага",
        )
        _ax.plot(
            _after_df[_x_column],
            _after_df[_y_column],
            marker="o",
            markersize=2.5,
            color="darkorange",
            label="после шага",
        )
        _ax.annotate(
            "",
            xy=(selected_step_row[_x_column], selected_step_row[_y_column]),
            xytext=(_previous_row[_x_column], _previous_row[_y_column]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "crimson",
                "linewidth": 2.4,
                "mutation_scale": 16,
            },
        )
        _ax.scatter(
            _previous_row[_x_column],
            _previous_row[_y_column],
            facecolors="white",
            edgecolors="crimson",
            linewidths=1.8,
            s=52,
            zorder=3,
            label="точка до",
        )
        _ax.scatter(
            selected_step_row[_x_column],
            selected_step_row[_y_column],
            color="crimson",
            marker="X",
            s=62,
            zorder=3,
            label="точка после",
        )
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.set_title(
            f"{_object_name}: выбранный шаг = "
            f"{selected_step_row[_step_column]:.3f}"
        )
        _ax.set_xlim(_x_min - _x_padding, _x_max + _x_padding)
        _ax.set_ylim(_y_min - _y_padding, _y_max + _y_padding)
        _ax.set_aspect("equal", adjustable="box")
        _ax.grid(alpha=0.25)
        _ax.legend()

    _local_step_ax.plot(
        local_step_df["time"],
        local_step_df["step_object1"],
        marker="o",
        markersize=3,
        label="объект 1",
    )
    _local_step_ax.plot(
        local_step_df["time"],
        local_step_df["step_object2"],
        marker="o",
        markersize=3,
        label="объект 2",
    )
    _local_step_ax.axvline(
        _candidate_time,
        color="crimson",
        linestyle="--",
        linewidth=1.3,
        label="проверяемая точка",
    )
    _local_step_ax.scatter(
        [_candidate_time, _candidate_time],
        [
            selected_step_row["step_object1"],
            selected_step_row["step_object2"],
        ],
        color="crimson",
        marker="X",
        s=52,
        zorder=3,
    )
    _local_step_ax.set_ylim(0, step_df["spatial_step"].max() * 1.05)
    _local_step_ax.set_xlabel("time")
    _local_step_ax.set_ylabel("пространственный шаг")
    _local_step_ax.set_title(
        "Локальные шаги: одинаковый масштаб для всех кандидатов"
    )
    _local_step_ax.grid(alpha=0.25)
    _local_step_ax.legend()

    _local_fig.suptitle(
        f"Локальное движение около time = {_candidate_time:.3f}"
    )
    _local_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Разбиение на непрерывные сегменты

    Локальная проверка показала, что первые десять крупнейших пространственных
    шагов — настоящие переходы между разными участками записи, а начиная с
    одиннадцатого движение остаётся непрерывным.

    Чтобы не задавать порог на глаз, ниже ищем самый большой **относительный
    разрыв** между соседними значениями в отсортированном `spatial_step`. Порог
    помещаем внутрь найденного пустого интервала. Строка после каждого скачка
    получает новый `segment_id` и становится первой строкой нового сегмента.
    """)
    return


@app.cell
def _(np, step_df):
    _sorted_steps = step_df["spatial_step"].dropna().sort_values(ascending=False)
    _neighbor_ratios = (
        _sorted_steps.iloc[:-1].to_numpy()
        / _sorted_steps.iloc[1:].to_numpy()
    )
    _gap_position = int(np.argmax(_neighbor_ratios))

    break_count = _gap_position + 1
    smallest_break_step = float(_sorted_steps.iloc[_gap_position])
    largest_regular_step = float(_sorted_steps.iloc[_gap_position + 1])
    spatial_break_threshold = float(
        np.sqrt(smallest_break_step * largest_regular_step)
    )

    segmented_df = step_df.copy()
    segmented_df["is_spatial_break"] = segmented_df["spatial_step"].gt(
        spatial_break_threshold
    )
    segmented_df["segment_id"] = (
        segmented_df["is_spatial_break"].cumsum().astype(int)
    )

    break_rows = segmented_df.loc[
        segmented_df["is_spatial_break"],
        ["time", "spatial_step", "step_object1", "step_object2", "segment_id"],
    ].copy()
    break_rows.index.name = "row_after_break"

    segment_summary = (
        segmented_df.groupby("segment_id", as_index=False)
        .agg(
            row_start=("time", lambda _series: int(_series.index[0])),
            row_end=("time", lambda _series: int(_series.index[-1])),
            row_count=("time", "size"),
            time_start=("time", "min"),
            time_end=("time", "max"),
        )
    )
    segment_summary["duration"] = (
        segment_summary["time_end"] - segment_summary["time_start"]
    )
    return (
        break_count,
        break_rows,
        largest_regular_step,
        segment_summary,
        segmented_df,
        smallest_break_step,
        spatial_break_threshold,
    )


@app.cell(hide_code=True)
def _(
    break_count,
    break_rows,
    largest_regular_step,
    mo,
    segment_summary,
    smallest_break_step,
    spatial_break_threshold,
):
    mo.vstack(
        [
            mo.md(f"""
            ### Вывод по разрывам

            По времени разрывов не обнаружено: `time` возрастает на всей записи,
            а редкие увеличенные `dt` не сопровождаются скачками координат.

            По пространственному шагу, наоборот, видны две чётко разделённые
            группы. Крупнейший обычный шаг равен `{largest_regular_step:.3f}`,
            самый маленький подтверждённый скачок —
            `{smallest_break_step:.3f}`, а между ними нет наблюдений. Поэтому
            значение `{spatial_break_threshold:.3f}`, расположенное внутри этого
            пустого интервала, используется как диагностический порог:
            `spatial_step > {spatial_break_threshold:.3f}` означает начало нового
            сегмента.

            Этот порог не является физической константой. Любое значение между
            `{largest_regular_step:.3f}` и `{smallest_break_step:.3f}` даст то же
            разбиение. В результате выделено **{break_count} переходов** и
            получено **{len(segment_summary)} непрерывных сегментов разной
            длины**.

            Дальнейшие графики, производные и модели нужно рассчитывать внутри
            каждого `segment_id`, не соединяя соседние сегменты через скачок.

            Ниже перечислены строки, с которых начинаются новые сегменты.
            """),
            break_rows,
            mo.md("### Размеры сегментов"),
            segment_summary,
        ]
    )
    return


@app.cell
def _(plt, segmented_df, segment_summary):
    _segment_fig, _segment_axes = plt.subplots(1, 2, figsize=(14, 6))
    _segment_colors = plt.get_cmap("tab20", len(segment_summary))

    for _segment_id, _segment in segmented_df.groupby("segment_id", sort=True):
        _color = _segment_colors(_segment_id)
        for _ax, _x_column, _y_column, _object_name in [
            (_segment_axes[0], "x1", "y1", "Объект 1"),
            (_segment_axes[1], "x2", "y2", "Объект 2"),
        ]:
            # Каждый segment_id рисуется отдельной линией: переходы между
            # сегментами намеренно не соединяются.
            _ax.plot(
                _segment[_x_column],
                _segment[_y_column],
                color=_color,
                linewidth=0.65,
                alpha=0.85,
                label=f"segment {_segment_id}",
            )
            _ax.scatter(
                _segment[_x_column].iloc[0],
                _segment[_y_column].iloc[0],
                color=_color,
                edgecolors="black",
                linewidths=0.35,
                s=22,
                zorder=3,
            )
            _ax.set_xlabel("x")
            _ax.set_ylabel("y")
            _ax.set_title(_object_name)
            _ax.set_aspect("equal", adjustable="box")
            _ax.grid(alpha=0.25)

    _handles, _labels = _segment_axes[1].get_legend_handles_labels()
    _segment_fig.legend(
        _handles,
        _labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Непрерывные сегменты",
    )
    _segment_fig.suptitle(
        "Траектории после сегментации: линии не проходят через разрывы"
    )
    _segment_fig.tight_layout(rect=(0, 0, 0.88, 1))

    _segment_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 1 — загрузка и аудит данных

    - [x] Загрузить `data.csv` из локального архива в `data/`.
    - [x] Удалить полностью пустые столбцы.
    - [x] Проверить типы, пропуски, дубликаты и порядок по `time`.
    - [x] Подтвердить диапазон и неравномерность временных шагов.
    - [x] Не приписывать физический смысл `distance`, `angle1`, `angle2`.

    **Ожидаемый артефакт:** короткая таблица data dictionary и две-три проверки
    качества данных.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 2 — траектории и сбросы

    - [x] Построить `x1`–`y1` и `x2`–`y2`.
    - [x] Построить все четыре координаты от времени.
    - [x] Проверить возможные разрывы по `time`.
    - [x] Рассчитать пространственный шаг каждого объекта.
    - [x] Выбрать порог разрыва и создать `segment_id`.
    - [x] Проверить, что графики не соединяют разные сегменты.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 3 — скорости и фазовые проекции

    - [ ] На одном чистом сегменте вычислить скорости по реальному `time`.
    - [ ] Сравнить центральные разности и Savitzky–Golay.
    - [ ] Построить скорость от времени.
    - [ ] Построить `x`–`v_x`, `y`–`v_y`, центр и относительное движение.

    **Стоп-критерий:** производные выглядят физически осмысленно и не содержат
    очевидных выбросов на границах сегментов.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 4 — первый SINDy MVP

    - [ ] Создать train/validation/test из разных `segment_id`.
    - [ ] Начать с состояния `(x1, y1, x2, y2)`.
    - [ ] Использовать полиномиальную библиотеку второй степени.
    - [ ] Подобрать разреживание на validation, не на test.
    - [ ] Выписать ненулевые члены найденных уравнений.

    **Важно:** не интерпретировать формулу как фундаментальный закон без проверки
    rollout и устойчивости.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO 5 — симуляция и вывод

    - [ ] Проинтегрировать модель из начального состояния test-сегмента.
    - [ ] Сравнить короткий rollout с настоящей траекторией.
    - [ ] Сравнить фазовые проекции и относительное движение.
    - [ ] Повторить главный результат с немного другим сглаживанием.
    - [ ] Сформулировать итог и ограничения модели.
    """)
    return


if __name__ == "__main__":
    app.run()
