import pandas as pd
import sidetable as stb


# TODO: add pandera. El shape de ids es 1351
def procedimientos_con_empresas_fantasma(df: pd.DataFrame, fantasma: pd.DataFrame):
    """Regresa una Dataframe con el id_unico de procedimiento y una columna
    que indica si el procedimiento fue con una empresa fantasma"""
    data = df.loc[df.Resultado == "ADJUDICADA"]
    # por nombre
    df_procs = data.loc[:, ["id_unico", "empresa_ganadora", "RFC"]].drop_duplicates()
    in_fantasma_empresa = df_procs.empresa_ganadora.isin(fantasma.empresa_fantasma)
    in_fantasma_rfc = df_procs.RFC.isin(fantasma.RFC)
    con_fantasma = in_fantasma_empresa | in_fantasma_rfc
    df_procs = df_procs.assign(con_empresa_fantasma=con_fantasma.astype(int))
    df_procs = df_procs.drop(["empresa_ganadora", "RFC"], axis=1).drop_duplicates()
    df_procs = df_procs.groupby("id_unico", as_index=False).con_empresa_fantasma.sum()
    df_procs = df_procs.assign(
        con_empresa_fantasma=df_procs.con_empresa_fantasma.astype(bool).astype(int)
    )
    return df_procs


def procedimientos_con_empresas_sancionadas(
    df: pd.DataFrame, sancionados: pd.DataFrame
):
    """Regresa una Dataframe con el id_unico y una columna que indica si
    el procedimiento fue con una empresa sancionada"""
    data = df.loc[df.Resultado == "ADJUDICADA"]
    df_procs = data.loc[:, ["id_unico", "empresa_ganadora"]].drop_duplicates()
    in_sancionadas = df_procs.empresa_ganadora.isin(sancionados.empresa_sancionada)
    df_procs = df_procs.assign(con_empresa_sancionada=in_sancionadas.astype(int))
    df_procs = df_procs.drop("empresa_ganadora", axis=1).drop_duplicates()
    df_procs = df_procs.groupby("id_unico", as_index=False).con_empresa_sancionada.sum()
    df_procs = df_procs.assign(
        con_empresa_sancionada=df_procs.con_empresa_sancionada.astype(bool).astype(int)
    )
    return df_procs


def documentacion_faltante_concursos_abiertos(
    df_archivos: pd.DataFrame, df_data: pd.DataFrame
):
    cols = [
        "publicacion",
        "preguntas_aclaraciones",
        "entrega_propuesta",
        "asignacion",
        "diferimiento",
    ]
    df_conteo = (
        df_archivos.set_index("id_unico")
        .loc[:, cols]
        .astype(bool)
        .astype(int)
        .sum(axis=1)
        .divide(5)
        .reset_index()
    )
    df_conteo = df_conteo.rename(columns={0: "pc_documentacion"})
    df_conteo = df_conteo.assign(pc_documentacion=df_conteo.pc_documentacion * 100)
    cond = (df_data.Resultado == "ADJUDICADA") & (
        df_data.tipo_contratacion == "concurso_abierto"
    )
    cols = ["id_unico", "empresa_productiva", "tipo_iniciativa"]
    data = df_data.loc[cond].groupby(cols, as_index=False).montos_maximos_mxn.sum()
    print(data.shape, data.id_unico.nunique())
    data = pd.merge(data, df_conteo, on="id_unico", how="left")
    data = data.rename(columns={"montos_maximos_mxn": "monto"})
    return data


def dias_entre_etapas(df: pd.DataFrame) -> pd.DataFrame:
    data = df.loc[df.Resultado == "ADJUDICADA"]
    df_ids = data.loc[:, ["id_unico"]].drop_duplicates()
    # no se calcula para adjudicaciones
    data = data.loc[data.tipo_contratacion.isin({"invitacion", "concurso_abierto"})]
    cols = ["id_unico", "tipo_contratacion", "fecha_recepcion_propuestas"]
    cols_publicado = cols + ["publicado"]
    cols_fallo = cols + ["fecha_fallo"]
    data_publicado = data.groupby(
        cols_publicado, as_index=False
    ).montos_maximos_mxn.sum()
    data_fallo = data.groupby(cols_fallo, as_index=False).montos_maximos_mxn.sum()
    diff = (
        data_publicado.fecha_recepcion_propuestas - data_publicado.publicado
    ).dt.days
    data_publicado = data_publicado.assign(dias_convocatoria_y_propuestas=diff).drop(
        "fecha_recepcion_propuestas", axis=1
    )
    diff = (data_fallo.fecha_fallo - data_fallo.fecha_recepcion_propuestas).dt.days
    data_fallo = data_fallo.assign(dias_propuestas_y_resultado=diff).drop(
        ["montos_maximos_mxn"], axis=1
    )
    data = pd.merge(
        data_publicado, data_fallo, how="outer", on=["id_unico", "tipo_contratacion"]
    )
    data = pd.merge(df_ids, data, on="id_unico", how="left")
    cols_drop = [
        "tipo_contratacion",
        "publicado",
        "montos_maximos_mxn",
        "fecha_recepcion_propuestas",
        "fecha_fallo",
    ]
    data = data.drop(cols_drop, axis=1)
    return data


# Features por proveedor


def dias_entre_fallo_y_rfc_empresa(df: pd.DataFrame) -> pd.DataFrame:
    data = df.loc[df.Resultado == "ADJUDICADA"]
    df_proveedores = data.loc[:, ["empresa_ganadora"]].drop_duplicates()
    cols = ["RFC", "empresa_ganadora", "fecha_creacion_empresa_rfc"]
    data = (
        data.groupby(cols, as_index=False)
        .fecha_fallo.min()
        .rename(columns={"fecha_fallo": "fecha_fallo_min"})
    )
    dias_entre_incorporacion_y_fallo = (
        data.fecha_fallo_min - data.fecha_creacion_empresa_rfc
    ).dt.days
    data = data.assign(
        dias_entre_incorporacion_y_fallo=dias_entre_incorporacion_y_fallo
    )
    data = pd.merge(df_proveedores, data, on="empresa_ganadora", how="left")
    data = data.sort_values("dias_entre_incorporacion_y_fallo")
    data = data.drop(["RFC", "fecha_creacion_empresa_rfc", "fecha_fallo_min"], axis=1)
    return data


def empresas_con_adjudicaciones(df: pd.DataFrame) -> pd.DataFrame:
    data = df.loc[df.Resultado == "ADJUDICADA"]
    df_proveedores = data.loc[:, ["empresa_ganadora"]].drop_duplicates()
    data = data.loc[data.montos_maximos_mxn > 0]
    df_adjs = (
        data.loc[data.tipo_contratacion == "adjudicacion"]
        .groupby("empresa_ganadora")
        .agg({"id_unico": "nunique", "montos_maximos_mxn": "sum"})
        .reset_index()
        .rename(
            columns={
                "id_unico": "adjudicaciones",
                "montos_maximos_mxn": "monto_adjudicaciones",
            }
        )
    )
    data = (
        data.groupby("empresa_ganadora")
        .agg({"id_unico": "nunique", "montos_maximos_mxn": "sum"})
        .reset_index()
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
    )
    data = pd.merge(data, df_adjs, on="empresa_ganadora", how="left")
    data = data.assign(
        adjudicaciones=data.adjudicaciones.fillna(0),
        monto_adjudicaciones=data.monto_adjudicaciones.fillna(0),
    ).drop(["contrataciones", "monto"], axis=1)
    data = pd.merge(df_proveedores, data, on="empresa_ganadora", how="left")
    return data


def market_share_tipo_iniciativa(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el market share para cada empresa en los diferentes tipos
    de suministros como Servicios, Obra pública y Bienes"""
    data = df.loc[df.Resultado == "ADJUDICADA"]
    df_proveedores = data.loc[:, ["empresa_ganadora"]].drop_duplicates()
    data = data.loc[data.montos_maximos_mxn > 0]
    data = (
        data.groupby(["empresa_ganadora", "tipo_iniciativa"])
        .agg({"id_unico": "nunique", "montos_maximos_mxn": "sum"})
        .rename(columns={"id_unico": "contrataciones", "montos_maximos_mxn": "monto"})
        .reset_index()
    )
    data = data.loc[data.tipo_iniciativa.isin({"Bienes", "Servicios", "Obra pública"})]
    # return data
    # calcular los porcentajes
    column = "monto"
    dfs = []
    for t in ["Bienes", "Servicios", "Obra pública"]:
        col_title = "_".join(t.split(" ")).lower()
        df_pcs = (
            data.loc[data.tipo_iniciativa == t]
            .stb.freq(["empresa_ganadora"], value=column)
            .drop([f"Cumulative {column}", "Cumulative Percent", column], axis=1)
            .set_index("empresa_ganadora")
        )
        df_pcs = df_pcs.assign(Percent=df_pcs.Percent * 100).rename(
            columns={"Percent": f"pc_{column}_{col_title}"}
        )
        dfs.append(df_pcs)
    df_market = pd.concat(dfs, axis=1).fillna(0)
    df_market = df_market.reset_index().rename(columns={"index": "empresa_ganadora"})
    df_proveedores = pd.merge(
        df_proveedores, df_market, on="empresa_ganadora", how="left"
    )
    return df_proveedores
