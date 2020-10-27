"""Funciones para generar las gráficas del análisis exploratorio
"""

import itertools
import pandas as pd
import holoviews as hv
import numpy as np
from datetime import date

EMPRESAS_IRRELEVANTES = {"PFE", "PPS", "PEE"}


def calcular_nhhi(df: pd.DataFrame, column: str) -> float:
    df_pcs = (
        df.stb.freq(["empresa_ganadora"], value=column)
        .drop([f"Cumulative {column}", "Cumulative Percent", column], axis=1)
        .rename(columns={"Percent": f"pc_{column}"})
    )
    n_empresas = df_pcs.empresa_ganadora.nunique()
    hhi = df_pcs[f"pc_{column}"].pow(2).sum()
    nhhi = (hhi - (1 / n_empresas)) / (1 - (1 / n_empresas))
    return nhhi


# TODO: add test
def normalized_herfindahl_hirschman_index(df: pd.DataFrame, value: str):
    valid = {"monto", "contrataciones"}
    if value not in valid:
        raise ValueError(f"{value} is not in {valid}")
    cols = ["empresa_productiva", "empresa_ganadora"]
    data = df.loc[df.Resultado == "ADJUDICADA"]
    data = (
        data.groupby(cols)
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    empresas = [
        e for e in data.empresa_productiva.unique() if e not in EMPRESAS_IRRELEVANTES
    ]
    # title = 'Índice Herfindahl–Hirschman normalizado'
    if value == "monto":
        title = "IHH normalizado por monto total"
    else:
        title = "IHH normalizado por número de contrataciones"
    nhhis = []
    for e in empresas:
        df_empresa = data.loc[data.empresa_productiva == e]
        if df_empresa.shape[0] == 0:
            continue
        nhhi = calcular_nhhi(df_empresa, column=value)
        nhhis.append((e, nhhi))
    bars = hv.Bars(nhhis, kdims="empresa productiva", vdims="nhhi", label=title)
    bars = bars.opts(width=450, height=300, toolbar=None)
    return bars


def money_ticks(min_val: int, max_val: int, n_ticks: int = 10):
    values = np.linspace(min_val, max_val, n_ticks)
    labels = [(v, f"${int(v):,}") for v in values]
    return labels


def porcentaje_contrataciones_y_monto_empresa_productiva(df: pd.DataFrame, value: str):
    valid = {"monto", "contrataciones"}
    if value not in valid:
        raise ValueError(f"{value} is not in {valid}")
    data = df.loc[df.Resultado == "ADJUDICADA"]
    data = (
        data.groupby(["empresa_productiva", "tipo_contratacion"])
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    empresas = {
        e for e in data.empresa_productiva.unique() if e not in EMPRESAS_IRRELEVANTES
    }
    data = data.loc[~data.empresa_productiva.isin(EMPRESAS_IRRELEVANTES)]
    dfs = []
    for v in ["contrataciones", "monto"]:
        dfs_empresas = []
        for e in empresas:
            df_empresa = data.loc[data.empresa_productiva == e]
            if df_empresa.shape[0] == 0:
                continue
            df_aux = (
                df_empresa.stb.freq(["tipo_contratacion"], value=v)
                .drop([f"Cumulative {v}", "Cumulative Percent", v], axis=1)
                .rename(columns={"Percent": f"pc_{v}"})
            )
            df_aux = df_aux.assign(empresa_productiva=e)
            dfs_empresas.append(df_aux)
        df_pcs = pd.concat(dfs_empresas, axis=0, ignore_index=True)
        dfs.append(df_pcs)

    df_join = pd.merge(dfs[0], dfs[1], on=["empresa_productiva", "tipo_contratacion"])
    df_join = df_join.sort_values(["empresa_productiva", "tipo_contratacion"])
    df_join = df_join.assign(
        pc_monto=df_join.pc_monto * 100,
        pc_contrataciones=df_join.pc_contrataciones * 100,
    )
    options = {
        "width": 500,
        "height": 300,
        "stacked": True,
        "toolbar": None,
        "legend_position": "right",
        "cmap": "Set2",
        "xlabel": "",
        "ylabel": "porcentaje (%)",
    }
    if value == "monto":
        label = f"Distribución del monto total"
    else:
        label = f"Distribución del número de contrataciones"
    bars = hv.Bars(
        df_join,
        kdims=["empresa_productiva", "tipo_contratacion"],
        vdims=f"pc_{value}",
        label=label,
    )
    bars = bars.opts(**options)
    return bars


def ganadores_por_tipo_iniciativa_y_emprepsa_productiva(df: pd.DataFrame):
    # TODO: posiblemente separar en muchas gráficas
    data = df.loc[df.Resultado == "ADJUDICADA"]
    data = (
        data.groupby(["empresa_productiva", "tipo_iniciativa"])
        .agg({"empresa_ganadora": "nunique", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "empresa_ganadora": "ganadores"})
    )
    plots = []
    empresas = [
        e for e in data.empresa_productiva.unique() if e not in EMPRESAS_IRRELEVANTES
    ]
    for emp in empresas:
        df_contrataciones = pd.melt(
            data.loc[data.empresa_productiva == emp],
            id_vars=["tipo_iniciativa"],
            value_vars=["contrataciones"],
        )
        df_ganadores = pd.melt(
            data.loc[data.empresa_productiva == emp],
            id_vars=["tipo_iniciativa"],
            value_vars=["ganadores"],
        )
        dfs = pd.concat([df_ganadores, df_contrataciones], axis=0, ignore_index=True)

        bars = hv.Bars(
            dfs, kdims=["tipo_iniciativa", "variable"], vdims="value", label=emp
        )
        bars = bars.sort()
        bars = bars.opts(width=850, height=270, toolbar=None, ylabel="", xlabel="")
        plots.append(bars)
    label = "Número de contrataciones y ganadores únicos por empresa productiva"
    layout = hv.Layout(plots).opts(
        hv.opts.Layout(toolbar=None, shared_axes=False, title=label)
    )
    layout = layout.cols(1)
    return layout


def dias_fallo_propuestas_tipo_contratacion(df: pd.DataFrame):
    cols = [
        "montos_maximos_mxn",
        "tipo_contratacion",
        "fecha_recepcion_propuestas",
        "fecha_fallo",
    ]
    cond = (df.tipo_contratacion != "adjudicacion") & (df.Resultado == "ADJUDICADA")
    data = df.loc[cond, cols]
    delta_dias = data.fecha_fallo - data.fecha_recepcion_propuestas
    data = data.assign(delta_dias=delta_dias.dt.days)
    histograms = []
    for t, n_bins in [("concurso_abierto", 35), ("invitacion", 20)]:
        data_tipo = data.loc[data.tipo_contratacion == t, ["delta_dias"]].dropna()
        freqs, edges = np.histogram(data_tipo.to_numpy(), n_bins, density=True)
        h = hv.Histogram(
            (edges, freqs),
            kdims="días",
            vdims="Frecuencia normalizada",
            label=" ".join(t.split("_")),
        )
        histograms.append(h)
    layout = hv.Layout(histograms)
    layout = layout.opts(
        hv.opts.Layout(
            title="Días entre la fecha de fallo y el envío de propuestas",
            shared_axes=True,
            toolbar=None,
        ),
    )
    return layout


def box_plot_monto(df: pd.DataFrame, column: str):
    valid = {
        "tipo_contratacion",
        "tipo_evento",
        "tipo_iniciativa",
        "empresa_productiva",
    }
    if column not in valid:
        m = f"column no puede tomar el valor {column}"
        raise ValueError(m)
    data = df.loc[df.Resultado == "ADJUDICADA"].rename(
        columns={"montos_maximos_mxn": "monto"}
    )
    # FIXME: algunos casos no sé porque tienen monto igual a cero
    data = data.loc[data.monto > 0]
    data = data.assign(monto_en_millones=data.monto.divide(1e6))
    max_val = data.monto_en_millones.max()
    yticks = money_ticks(0, max_val, 7)
    box = hv.BoxWhisker(data, kdims=column, vdims=["monto_en_millones"])
    box = box.opts(
        width=700,
        height=400,
        cmap="Set1",
        xlabel="",
        ylabel="",
        toolbar=None,
        yticks=yticks,
        title="Monto en millones de pesos",
    )
    return box


def heatmap_por_tipo(df: pd.DataFrame, col_x: str, col_y: str, value: str):
    valid = {"monto", "contrataciones"}
    if value not in valid:
        raise ValueError(f"{value} is not in {valid}")
    data = (
        df.loc[df.Resultado == "ADJUDICADA"]
        .groupby([col_x, col_y])
        .agg({"id_unico": "nunique", "montos_maximos_mxn": "sum"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    data = data.assign(monto_en_millones=data.monto.divide(1e6))
    contrataciones_label = data.contrataciones.map(lambda c: f"{c:,}")
    monto_label = data.monto_en_millones.map(lambda c: f"${c:,.2f}")
    data = data.assign(
        contrataciones_label=contrataciones_label, monto_en_millones_label=monto_label
    )
    options = {
        "width": 750,
        "height": 250,
        "cmap": "viridis",
        "colorbar": True,
        "toolbar": None,
        "xlabel": "",
        "ylabel": "",
        "colorbar_opts": {"major_label_text_align": "left"},
    }
    if value == "monto":
        value = "monto_en_millones"
        label = "Monto en millones de pesos"
        text_font_size = "8pt"
    else:
        label = "Número de contrataciones"
        text_font_size = "9pt"
    hm = hv.HeatMap(data, kdims=[col_x, col_y], vdims=value, label=label)
    hm = hm.opts(**options)
    labels = hv.Labels(data, kdims=[col_x, col_y], vdims=f"{value}_label", label=label)
    labels = labels.opts(text_font_size=text_font_size, toolbar=None)
    overlay = hm * labels
    return overlay


def scatter_contrataciones_y_montos_empresas(df: pd.DataFrame):
    cols = [
        "montos_maximos_mxn",
        "tipo_iniciativa",
        "tipo_contratacion",
        "empresa_ganadora",
        "id_unico",
    ]
    data = (
        df.loc[df.Resultado == "ADJUDICADA", cols]
        .groupby(["empresa_ganadora", "tipo_iniciativa", "tipo_contratacion"])
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones"})
    )
    # TODO: hay dos casos en la que los montos son cero, se deben verificar
    data = data.loc[data.montos_maximos_mxn > 0]
    iniciativas = data.tipo_iniciativa.unique()
    contrataciones = data.tipo_contratacion.unique()
    scatters = {}
    options = {
        "tools": ["hover"],
        "width": 500,
        "height": 400,
        "size": 7,
        "alpha": 0.5,
        "logy": True,
        "logx": True,
    }
    for iniciativa, contratacion in itertools.product(iniciativas, contrataciones):
        cond = (data.tipo_iniciativa == iniciativa) & (
            data.tipo_contratacion == contratacion
        )
        selected = data.loc[cond]
        scatter = hv.Scatter(
            selected,
            kdims="contrataciones",
            vdims=["montos_maximos_mxn", "empresa_ganadora"],
        )
        scatters[(iniciativa, contratacion)] = scatter.opts(**options)
    holomap = hv.HoloMap(
        scatters,
        kdims=["tipo_iniciativa", "tipo_contratacion"],
        label="número de contrataciones vs montos",
    )
    grid = hv.GridSpace(holomap,)
    grid = grid.opts(hv.opts.GridSpace(plot_size=150))
    return grid


def top_empresas_gandoras(df: pd.DataFrame, value: str, n: int = 20):
    valid = {"monto", "contrataciones"}
    if value not in valid:
        raise ValueError(f"{value} is not in {valid}")
    data = (
        df.loc[df.Resultado == "ADJUDICADA"]
        .groupby(["empresa_ganadora"])
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    data = data.assign(monto_en_millones=data.monto.divide(1e6)).drop("monto", axis=1)
    contrataciones_label = data.contrataciones.map(lambda c: f"{c:,}")
    monto_label = data.monto_en_millones.map(lambda c: f"${c:,.2f}")
    data = data.assign(
        contrataciones_label=contrataciones_label, monto_en_millones_label=monto_label
    )
    width = 1200
    height = 800
    if value == "monto":
        value = "monto_en_millones"
        title = f"Top {n} empresas por monto en millones de pesos"
        offset = 150
    else:
        title = f"Top {n} empresas por número de contrataciones"
        offset = 0.3
    df_top = data.nlargest(n, columns=value).sort_values(value, ascending=False)
    bars = hv.Bars(df_top, kdims="empresa_ganadora", vdims=value)
    bars = bars.opts(
        xrotation=78,
        toolbar=None,
        xlabel="",
        ylabel="",
        width=width,
        height=height,
        title=title,
    )
    if value == "monto_en_millones":
        yticks = money_ticks(0, df_top.monto_en_millones.max(), 6)
        bars = bars.opts(yticks=yticks)
    labels = hv.Labels(
        df_top, kdims=["empresa_ganadora", value], vdims=f"{value}_label"
    )
    labels = labels.opts(
        yoffset=offset,
        width=width,
        height=height,
        toolbar=None,
        text_font_size="9pt",
        xlabel="",
        ylabel="",
    )
    overlay = bars * labels
    return overlay


def top_empresas_gandoras_por_iniciativa(df: pd.DataFrame, column: str, n: int = 20):
    valid = {"montos_maximos_mxn", "id_unico"}
    if column not in valid:
        m = "columna no válida"
        raise ValueError(m)
    if column == "montos_maximos_mxn":
        title = f"Top {n} de empresas con mayor monto en millones de pesos"
        column = "monto_en_millones"
    else:
        title = f"Top {n} de empresas con mayor número de contrataciones"
        column = "contrataciones"
    data = (
        df.loc[df.Resultado == "ADJUDICADA"]
        .groupby(["empresa_ganadora", "tipo_iniciativa"])
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    data = data.assign(monto_en_millones=data.monto.divide(1e6)).drop("monto", axis=1)
    iniciativas = sorted(data.tipo_iniciativa.unique(), reverse=True)
    plots = []
    for iniciativa in iniciativas:
        df_aux = (
            data.loc[data.tipo_iniciativa == iniciativa]
            .nlargest(n, columns=column)
            .sort_values(column)
        )
        xticks = money_ticks(0, df_aux.monto_en_millones.max(), 6)
        bars = hv.Bars(df_aux, kdims="empresa_ganadora", vdims=column, label=iniciativa)
        bars = bars.opts(
            toolbar=None, invert_axes=True, xlabel="", ylabel="", width=800, height=300
        )
        if column == "monto_en_millones":
            bars = bars.opts(xticks=xticks)
        plots.append(bars)
    layout = hv.Layout(plots).cols(1)
    layout = layout.opts(hv.opts.Layout(title=title, shared_axes=False, toolbar=None))
    return layout


def cumplimiento_documentos_concursos_abiertos(df: pd.DataFrame):
    cols = [
        "publicacion",
        "preguntas_aclaraciones",
        "entrega_propuesta",
        "asignacion",
        "diferimiento",
    ]
    n_concursos = df.id_unico.nunique()
    df_conteo = df.set_index("id_unico").loc[:, cols].astype(bool).astype(int).sum()
    df_pcs = (
        (df_conteo * 100)
        .divide(n_concursos)
        .reindex(cols)
        .reset_index()
        .rename(columns={"index": "etapa", 0: "pc"})
    )
    pc_label = df_pcs.pc.map(lambda c: f"{c:,.2f}")
    df_pcs = df_pcs.assign(pc_label=pc_label)
    title = "Cumplimiento de la documentación por etapa en concursos abiertos"
    bars = hv.Bars(df_pcs, kdims="etapa", vdims="pc")
    bars = bars.opts(
        toolbar=None,
        invert_axes=False,
        xlabel="",
        ylabel="porcentaje (%)",
        width=600,
        height=300,
        title=title,
    )
    labels = hv.Labels(df_pcs, kdims=["etapa", "pc"], vdims="pc_label")
    labels = labels.opts(text_font_size="8pt", yoffset=2)
    overlay = bars * labels
    return overlay


def montos_por_participantes_y_tipo(df: pd.DataFrame, tipo_contratacion: str):
    cond = (df.tipo_contratacion == tipo_contratacion) & (df.Resultado == "ADJUDICADA")
    data = df.loc[cond]
    cols = [
        "id_unico",
        "empresa_1",
        "empresa_2",
        "empresa_3",
        "empresa_4",
        "empresa_5",
        "empresa_6",
        "empresa_7",
    ]
    participantes = data.loc[:, cols].drop_duplicates().set_index("id_unico")
    participantes = (
        participantes.apply(lambda p: ~p.isna(), axis=1)
        .sum(axis=1)
        .reset_index()
        .rename(columns={0: "num_participantes"})
    )
    montos_tipos = (
        data.groupby("id_unico", as_index=True)
        .agg({"montos_maximos_mxn": "sum", "tipo_iniciativa": "unique"})
        .reset_index()
    )
    montos_tipos = montos_tipos.assign(
        tipo_iniciativa=montos_tipos.tipo_iniciativa.map(lambda l: l[0])
    )
    participantes = pd.merge(participantes, montos_tipos, on="id_unico")
    scatter = hv.Scatter(
        participantes,
        kdims="num_participantes",
        vdims=["montos_maximos_mxn", "tipo_iniciativa"],
    )
    scatter = scatter.opts(
        width=600,
        height=450,
        alpha=0.6,
        color="tipo_iniciativa",
        cmap="Set1",
        size=8,
        logy=True,
        xticks=8,
        tools=["hover"],
        title=tipo_contratacion,
    )
    return scatter


def agregaciones_por_tipo(df: pd.DataFrame, column: str, value: str):
    valid = {
        "tipo_contratacion",
        "tipo_evento",
        "tipo_iniciativa",
        "empresa_productiva",
    }
    if column not in valid:
        m = f"column no puede tomar el valor {column}"
        raise ValueError(m)
    if value not in {"monto", "contrataciones"}:
        raise ValueError(f"{value} is not in monto or contrataciones")
    data = (
        df.loc[df.Resultado == "ADJUDICADA"]
        .groupby(column)
        .agg({"montos_maximos_mxn": "sum", "id_unico": "nunique"})
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
        .reset_index()
    )
    data = data.assign(monto_en_millones=data.monto.divide(1e6))
    contrataciones_label = data.contrataciones.map(lambda c: f"{c:,}")
    monto_label = data.monto_en_millones.map(lambda c: f"${c:,.2f}")
    data = data.assign(
        contrataciones_label=contrataciones_label, monto_en_millones_label=monto_label
    )
    if value == "monto":
        title = "Monto en millones de pesos"
        value = "monto_en_millones"
    else:
        title = "Número de contrataciones"
    bars = hv.Bars(data, kdims=column, vdims=value)
    labels = hv.Labels(data, kdims=[column, value], vdims=f"{value}_label")
    labels = labels.opts(yoffset=22, text_font_size="9pt", xlabel="", ylabel="")
    bars = bars.opts(
        title=f"{title} por {' '.join(column.split('_'))}",
        width=500,
        height=300,
        toolbar=None,
        xlabel="",
        ylabel="",
    )
    bars = bars * labels
    if value == "monto_en_millones":
        max_val = data.monto_en_millones.max()
        yticks = money_ticks(0, max_val, 6)
        bars = bars.opts(
            hv.opts.Bars(yticks=yticks),
            hv.opts.Labels(yoffset=1900, text_font_size="8pt"),
        )
    return bars


def serie_tiempo_monto_tipo_contratacion(df: pd.DataFrame):
    cols = ["publicado", "montos_maximos_mxn", "tipo_contratacion"]
    data = df.loc[df.Resultado == "ADJUDICADA", cols]
    month = [
        date(y, m, 1) for m, y in zip(data.publicado.dt.month, data.publicado.dt.year)
    ]
    data = (
        data.assign(month=month)
        .groupby(["month", "tipo_contratacion"], as_index=False)
        .sum()
        .pivot(index="month", columns="tipo_contratacion", values="montos_maximos_mxn")
        # .fillna(0)
        .reset_index()
    )
    # TODO: simplificar y poner el scatter
    ts_adj = hv.Curve(data, kdims="month", vdims="adjudicacion", label="Adjudicaciones")
    ts_concurso = hv.Curve(
        data, kdims="month", vdims="concurso_abierto", label="Concursos abiertos"
    )
    ts_invitacion = hv.Curve(
        data, kdims="month", vdims="invitacion", label="Invitaciones"
    )
    # ts = hv.Area.stack(ts_adj * ts_concurso * ts_invitacion)
    ts = ts_adj * ts_concurso * ts_invitacion
    yticks = money_ticks(0, int(1.5e10), 8)
    ts = ts.opts(
        title="Montos por tipo de contratación", width=700, height=400, yticks=yticks
    )
    # ts = ts.opts(hv.opts.Layout(toolbar=None))
    return ts


def serie_tiempo_monto_contrataciones_por_tipo(df: pd.DataFrame):
    cols = ["publicado", "id_unico", "tipo_contratacion", "montos_maximos_mxn"]
    data = df.loc[df.Resultado == "ADJUDICADA", cols]
    month = [
        date(y, m, 1) for m, y in zip(data.publicado.dt.month, data.publicado.dt.year)
    ]
    data = (
        data.assign(month=month)
        .groupby(["month", "tipo_contratacion"], as_index=True)
        .agg({"id_unico": "nunique", "montos_maximos_mxn": "sum"})
        .reset_index()
    )
    vals_titles = [
        ("id_unico", "Número contrataciones por tipo de contratación"),
        ("montos_maximos_mxn", "Montos por tipo de contratación"),
    ]
    plots = []
    for val, title in vals_titles:
        ts_data = data.pivot(
            index="month", columns="tipo_contratacion", values=val
        ).reset_index()
        curves = []
        for tipo in ["adjudicacion", "concurso_abierto", "invitacion"]:
            label = " ".join(tipo.split(sep="_")).title()
            curve = hv.Curve(ts_data, kdims="month", vdims=tipo, label=label)
            curves.append(curve)
        ts = curves[0] * curves[1] * curves[2]
        ts = ts.opts(title=title, width=700, height=400, toolbar=None)
        plots.append(ts)
    layout = (plots[0] + plots[1]).opts(hv.opts.Layout(shared_axes=False, toolbar=None))
    layout = layout.cols(1)
    return layout


def dist_monto_tipo_iniciativa_y_contratacion(df: pd.DataFrame):
    # TODO: en lugar de poner los box plots juntos separarlos en mas graficas
    cols = ["montos_maximos_mxn", "tipo_iniciativa", "tipo_contratacion"]
    data = df.loc[df.Resultado == "ADJUDICADA", cols]
    box = hv.BoxWhisker(
        data,
        kdims=["tipo_contratacion", "tipo_iniciativa"],
        vdims=["montos_maximos_mxn"],
    )
    box = box.opts(
        width=1200,
        height=400,
        tools=["hover"],
        cmap="Set1",
        box_fill_color=hv.dim("tipo_iniciativa").str(),
    )
    return box
