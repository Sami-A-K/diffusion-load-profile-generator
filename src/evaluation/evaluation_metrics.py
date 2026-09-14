import numpy as np
import pandas as pd

"""
Pirmärmetrik: Dauerlinienfehler pro Zeitkontext
Über eine Jahresdauerlinie kann die Häufigkeitsverteilung der Last dargestellt werden. Die chronologische Zuordnung der Last geht dabei verloren. Um sowohl die Häufigkeitsverteilung als auch die zeitliche Verteilung zu Berücksichtigen, werden Dauerlinien pro Zeitkontext gebildet. Ein Zeitkontext soll dabei Perioden zusammenfassen, die ein ähnliches Lastverhalten aufweisen. Basis ist dafür die (alte) BDEW Charakterisierung bei Standardlastprofilen. Diese teilt Lastprofile in charakteristische Zeitzonen (Sommer, Winter, Übergang) und charakteristische Tage (Werktag, Samstag, Sonn-/Feiertag) auf. Zusätzlich werden sechs Tageszeiten definiert. Jede Last pro Stunde kann somit einem von 54 Zeitkontexten zugewiesen werden.
Pro Sensor werden alle realen und generierten Lasten den Zeitkontexten zugewiesen. Innherhalb aller Zeitkontexte werden Dauerlinien gebildet (absteigende Sortierung der Last). Die Fläche zwischen realer und generierter Dauerlinie ist der Fehler in der Häufigkeitsverteilung der Last in dem Zeitkontext. Dieser ist der Betrag der Differenz jedes Punktes der Dauerlinie. Die Fehler werden basierend auf der Anzahl der Lasten pro Zeitkontext gewichtet und gemittelt. Um ein Modell in der Qualität der Lastprofilgeneration zu bewerten, kann dieser Wert über alle Sensoren des Testdatensatzes gemittelt werden.
"""

SEASON_NAMES = ["Winter", "Uebergang", "Sommer"]
SEASON_BOUNDS = [(321, 514, 1), (515, 914, 2), (915, 1031, 1)]
TIME_OF_DAY_NAMES=["Morgen", "Vormittag", "Mittag", "Nachmittag", "Abend", "Nacht"]
TIME_OF_DAY_BOUNDS = [(6,9), (9,12), (12,14), (14,17), (17,22), (22,6)]
DAYTYPE_NAMES = ["Werktag", "Samstag", "Sonntag"]

N_CONTEXTS = len(SEASON_NAMES) * len(DAYTYPE_NAMES) * len(TIME_OF_DAY_NAMES)


def assign_context(y_real, y_gen, days, is_holiday) -> tuple[list[np.ndarray], list[np.ndarray]]:

    dt = pd.DatetimeIndex(days)
    month_day = dt.month.to_numpy() * 100 + dt.day.to_numpy()
    season = np.zeros(len(days), dtype=int)
    for start, end, idx in SEASON_BOUNDS:
        season[(month_day >= start) & (month_day <= end)] = idx

    is_saturday = dt.dayofweek.to_numpy() == 5
    daytype = np.where(np.asarray(is_holiday).astype(bool), 2, np.where(is_saturday, 1, 0))
    day_context = (season * len(DAYTYPE_NAMES) + daytype) * len(TIME_OF_DAY_NAMES)

    tod = np.full(24, -1, dtype=int)
    for i, (start, end) in enumerate(TIME_OF_DAY_BOUNDS):
        hours = range(start, end) if start < end else list(range(start, 24)) + list(range(0, end))
        for h in hours:
            tod[h] = i

    context_ids = (day_context[:, None] + tod[None, :]).reshape(-1)
    y_real_flat = y_real.reshape(-1)
    y_gen_flat = y_gen.reshape(-1)

    y_real_context = [y_real_flat[context_ids == c] for c in range(N_CONTEXTS)]
    y_gen_context = [y_gen_flat[context_ids == c] for c in range(N_CONTEXTS)]
    return y_real_context, y_gen_context


def assign_hour(y_real, y_gen) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Wie assign_context, aber mit der Stunde des Tages (0..23) als Gruppe.
    """
    n_days = y_real.shape[0]
    hour_ids = np.tile(np.arange(24), n_days)
    y_real_flat = y_real.reshape(-1)
    y_gen_flat = y_gen.reshape(-1)
    return ([y_real_flat[hour_ids == h] for h in range(24)],
            [y_gen_flat[hour_ids == h] for h in range(24)])


def duration_curve_error(y_real_groups: list[np.ndarray], y_gen_groups: list[np.ndarray]) -> np.ndarray:
    """
    Fehler zwischen realer und generierter Dauerlinie je Gruppe (Zeitkontext, Stunde-des-Tages
    oder eine einzelne globale Gruppe ueber alle Stunden eines Sensors)
    """
    errors = np.full(len(y_real_groups), np.nan)
    for c, (real_c, gen_c) in enumerate(zip(y_real_groups, y_gen_groups)):
        if len(real_c) == 0:
            continue
        duration_curve_real = np.sort(real_c)
        duration_curve_gen = np.sort(gen_c)
        errors[c] = np.mean(np.abs(duration_curve_real - duration_curve_gen))

    return errors


def duration_curve_error_per_context(y_real, y_gen, sensor_ids, days, is_holiday) -> np.ndarray:
    """
    Fehler pro Zeitkontext, erst je Sensor berechnet und dann über die Sensoren gemittelt.
    Leere Kontexte eines Sensors (NaN) bleiben unberücksichtigt.
    return array mit 54 werten, 1 wert pro zeitkontext
    """
    errors_per_sensor = []
    for sid in np.unique(sensor_ids):
        m = sensor_ids == sid
        y_real_context, y_gen_context = assign_context(y_real[m], y_gen[m], days[m], is_holiday[m])
        errors_per_sensor.append(duration_curve_error(y_real_context, y_gen_context))

    return np.nanmean(np.stack(errors_per_sensor), axis=0)


def duration_curve_error_model(y_real, y_gen, sensor_ids, days, is_holiday) -> float:
    """
    Eine Kennzahl für das Modell: je Sensor die 54 Kontextfehler nach Stundenzahl gewichtet
    mitteln, danach gleichgewichtet über die Sensoren mitteln.
    """
    sensor_errors = []
    for sid in np.unique(sensor_ids):
        m = sensor_ids == sid
        y_real_context, y_gen_context = assign_context(y_real[m], y_gen[m], days[m], is_holiday[m])
        errors = duration_curve_error(y_real_context, y_gen_context)
        counts = np.array([len(c) for c in y_real_context])
        filled = counts > 0
        if not filled.any():
            continue
        sensor_errors.append(np.sum(errors[filled] * counts[filled]) / counts[filled].sum())

    return float(np.mean(sensor_errors))


def duration_curve_error_per_sensor(y_real, y_gen, sensor_ids, days, is_holiday) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dauerlinienfehler je Sensor sowie je Sensor und Zeitkontext.

    global  - eine Dauerlinie über alle Stunden eines Sensors.
    context - Dauerlinienfehler je Zeitkontext, nach Stundenzahl gewichtet gemittelt.
    error_structure = context - global (context >= global).
    hour    - Dauerlinienfehler je Stunde des Tages, nach Stundenzahl gewichtet gemittelt.

    Returns: (per_sensor mit sensor_id/n_days/n_empty_contexts/error_global/error_context/
    error_structure/error_hour, per_context mit sensor_id/season/daytype/tod/n/error).
    """
    sensor_rows, context_rows = [], []
    n_daytype, n_tod = len(DAYTYPE_NAMES), len(TIME_OF_DAY_NAMES)
    for sid in np.unique(sensor_ids):
        m = sensor_ids == sid
        yr, yg = y_real[m], y_gen[m]
        n_days = int(m.sum())

        y_real_ctx, y_gen_ctx = assign_context(yr, yg, days[m], is_holiday[m])
        errors_ctx = duration_curve_error(y_real_ctx, y_gen_ctx)
        counts = np.array([len(c) for c in y_real_ctx])
        filled = counts > 0
        error_context = np.sum(errors_ctx[filled] * counts[filled]) / counts[filled].sum()

        error_global = duration_curve_error([yr.reshape(-1)], [yg.reshape(-1)])[0]

        y_real_hour, y_gen_hour = assign_hour(yr, yg)
        errors_hour = duration_curve_error(y_real_hour, y_gen_hour)
        hour_counts = np.array([len(c) for c in y_real_hour])
        error_hour = np.sum(errors_hour * hour_counts) / hour_counts.sum()

        sensor_rows.append({
            "sensor_id": sid, "n_days": n_days, "n_empty_contexts": int((~filled).sum()),
            "error_global": error_global, "error_context": error_context, "error_structure": error_context - error_global,
            "error_hour": error_hour,
        })
        for c in range(N_CONTEXTS):
            s, d, t = c // (n_daytype * n_tod), (c // n_tod) % n_daytype, c % n_tod
            context_rows.append({"sensor_id": sid, "season": SEASON_NAMES[s], "daytype": DAYTYPE_NAMES[d],
                                 "tod": TIME_OF_DAY_NAMES[t], "n": int(counts[c]), "error": errors_ctx[c]})
    return pd.DataFrame(sensor_rows), pd.DataFrame(context_rows)


CONTEXT_KEYS = ["season", "daytype", "tod"]


def order_contexts(per_context: pd.DataFrame) -> pd.DataFrame:
    """season/daytype/tod als geordnete Categoricals in der Reihenfolge der Konstanten oben."""
    out = per_context.copy()
    for col, names in zip(CONTEXT_KEYS, [SEASON_NAMES, DAYTYPE_NAMES, TIME_OF_DAY_NAMES]):
        out[col] = pd.Categorical(out[col], categories=names, ordered=True)
    return out


def context_error_over_sensors(per_context: pd.DataFrame) -> pd.DataFrame:
    """
    Fehler je Zeitkontext, gemittelt über die Sensoren (jeder Sensor gleich gewichtet, leere
    Kontexte ausgenommen).

    share_mean: Anteil des Kontexts an der stundengewichteten Fehlersumme eines Sensors, je
    Sensor gebildet und gemittelt (Summe über alle Kontexte 100 %).
    index_mean: Kontextfehler geteilt durch den error_context desselben Sensors, je Sensor
    gebildet und gemittelt (1.0 = durchschnittliche Abweichung dieses Gebäudes).

    per_context: zweiter Rückgabewert von duration_curve_error_per_sensor.
    Returns: eine Zeile je Zeitkontext mit n_sensors, n_hours, error_mean, error_median, error_p90,
    share_mean, n_hours_share (in Prozent), index_mean, index_median.
    """
    filled = order_contexts(per_context[per_context["n"] > 0]).copy()
    n_total = filled["sensor_id"].nunique()
    by_sensor = filled.groupby("sensor_id")
    filled["contrib"] = filled["n"] * filled["error"]
    filled["share"] = filled["contrib"] / by_sensor["contrib"].transform("sum")
    filled["hour_share"] = filled["n"] / by_sensor["n"].transform("sum")
    # Bezugsgröße des Index: error_context des Sensors
    filled["index"] = filled["error"] / (by_sensor["contrib"].transform("sum") / by_sensor["n"].transform("sum"))

    grouped = filled.groupby(CONTEXT_KEYS, observed=False)
    out = grouped.agg(
        n_sensors=("error", "size"),
        n_hours=("n", "sum"),
        error_mean=("error", "mean"),
        error_median=("error", "median"),
        error_p90=("error", lambda s: s.quantile(0.9)),
        share_sum=("share", "sum"),
        hour_share_sum=("hour_share", "sum"),
        index_mean=("index", "mean"),
        index_median=("index", "median"),
    ).reset_index()
    # Division durch die Gesamtzahl der Sensoren: fehlende Kontexte zählen als Anteil 0
    out["share_mean"] = out.pop("share_sum") / n_total * 100
    out["n_hours_share"] = out.pop("hour_share_sum") / n_total * 100
    return out.sort_values(CONTEXT_KEYS).reset_index(drop=True)


"""
Sekundärmetrik: Tagesgrenzen-Sprungfaktor
Durch die Tageweise generierung der Lastprofile mit anschließender Verkettung, ist die Last an den Tagesgrenzen eines Profils von besonderer Bedeutung. Durch die unabhängige Generierung können unrealistische Sprünge in der Last entstehen. Deshalb werden die Sprünge zwischen allen verketteten Tagesprofilen zwischen Stunde 23 und Stunde 0 ausgewertet und mit den echten Lastsprüngen des Sensors verglichen. Die Differenz in den Sprüngen wird über alle Sensoren des Testdatensatzes gemittelt.
"""

def day_boundary_error_per_sensor(y_real, y_gen, sensor_ids, dates) -> pd.DataFrame:
    """
    Mittlerer Sprung zwischen Stunde 23 eines Tages und Stunde 0 des Folgetags, je Sensor. Beruecksichtigt nur Uebergaenge zwischen kalendarisch aufeinanderfolgenden Tagen desselben
    Sensors.
    jump_factor = generiert / real; 1.0 = die generierten Tagesgrenzen sind so glatt wie die realen, Werte darueber = das Modell springt zu stark.
    """
    order = np.lexsort((dates, sensor_ids))
    s_sorted, d_sorted = sensor_ids[order], dates[order]
    yr_sorted, yg_sorted = y_real[order], y_gen[order]

    valid_transition = (s_sorted[1:] == s_sorted[:-1]) & ((d_sorted[1:] - d_sorted[:-1]) == np.timedelta64(1, "D"))
    real_jumps = np.abs(yr_sorted[1:, 0] - yr_sorted[:-1, 23])[valid_transition]
    gen_jumps = np.abs(yg_sorted[1:, 0] - yg_sorted[:-1, 23])[valid_transition]

    df = pd.DataFrame({"sensor_id": s_sorted[1:][valid_transition], "real_jump": real_jumps, "gen_jump": gen_jumps})
    per_sensor = df.groupby("sensor_id").agg(
        n_transitions=("real_jump", "size"),
        abs_jump_real_mean=("real_jump", "mean"),
        abs_jump_gen_mean=("gen_jump", "mean"),
    ).reset_index()
    per_sensor["jump_factor"] = per_sensor["abs_jump_gen_mean"] / per_sensor["abs_jump_real_mean"]
    return per_sensor
