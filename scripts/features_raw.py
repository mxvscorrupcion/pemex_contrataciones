import pandas as pd
from pemex_contratos.preprocess import read_lista_contribuyentes_69b

# Funciones para features de procedimientos
from pemex_contratos.features import procedimientos_con_empresas_fantasma
from pemex_contratos.features import procedimientos_con_empresas_sancionadas
from pemex_contratos.features import dias_entre_etapas

# Funciones para features de empresas
from pemex_contratos.features import dias_entre_fallo_y_rfc_empresa
from pemex_contratos.features import empresas_con_adjudicaciones
from pemex_contratos.features import market_share_tipo_iniciativa

if __name__ == "__main__":
    path_contrataciones = "../data/processed/contrataciones_pemex.csv"
    path_listado = "../data/raw/Listado_Completo_69-B.csv"
    path_sancionadas = "../data/processed/lista_empresas_sancionadas.csv"
    path_features_procs = "../data/processed/features_raw_procedimientos.csv"
    path_features_empresas = "../data/processed/features_raw_empresas.csv"
    df_data = pd.read_csv(
        path_contrataciones,
        dtype={"montos_maximos_mxn": float, "montos_minimos_mxn": float},
        parse_dates=[
            "publicado",
            "fecha_fallo",
            "fecha_recepcion_propuestas",
            "fecha_preguntas_y_respuestas",
            "fecha_creacion_empresa_rfc",
        ],
    )
    df_listado = read_lista_contribuyentes_69b(path_listado)
    df_sancionados = pd.read_csv(path_sancionadas)
    cond = df_data.Resultado == "ADJUDICADA"
    n_procedimientos = df_data.loc[cond].id_unico.nunique()
    n_empresas = df_data.loc[cond].empresa_ganadora.nunique()
    del cond
    # Features de procedimientos
    dfs_procs = [
        procedimientos_con_empresas_fantasma(df_data, df_listado).set_index("id_unico"),
        procedimientos_con_empresas_sancionadas(df_data, df_sancionados).set_index(
            "id_unico"
        ),
        dias_entre_etapas(df_data).set_index("id_unico"),
    ]
    df_features_procs = (
        pd.concat(dfs_procs, axis=1).reset_index().rename(columns={"index": "id_unico"})
    )
    assert df_features_procs.shape[0] == n_procedimientos
    # Features de empresas
    dfs_empresas = [
        dias_entre_fallo_y_rfc_empresa(df_data).set_index("empresa_ganadora"),
        empresas_con_adjudicaciones(df_data).set_index("empresa_ganadora"),
        market_share_tipo_iniciativa(df_data).set_index("empresa_ganadora"),
    ]
    df_features_empresas = (
        pd.concat(dfs_empresas, axis=1)
        .reset_index()
        .rename(columns={"index": "empresa_ganadora"})
    )
    assert df_features_empresas.shape[0] == n_empresas
    df_features_procs.to_csv(
        path_features_procs, index=False, quoting=1, encoding="utf-8"
    )
    df_features_empresas.to_csv(
        path_features_empresas, index=False, quoting=1, encoding="utf-8"
    )
