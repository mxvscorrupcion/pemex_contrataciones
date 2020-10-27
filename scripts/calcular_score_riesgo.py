"""Script que genera la tabla de features binarios por contrato
y calcula el score de riesgo para cada uno de ellos"""
import numpy as np
import pandas as pd
from pemex_contratos.inai.load_data import cargar_todos_procedimientos
from pemex_contratos.inai.load_data import cargar_tabla_ofertas
from pemex_contratos.inai.load_data import cargar_tabla_cotizaciones
from pemex_contratos.inai.load_data import cargar_tabla_asistentes_junta_aclaraciones
from pemex_contratos.inai.load_data import cargar_tabla_posibles_contratantes
from pemex_contratos.inai.features_contrato import features_binarios_contratos
from pemex_contratos.inai.features_procedimiento import features_binarios_procedimientos
from pemex_contratos.inai.features_proveedor import features_binarios_proveedores
from pemex_contratos.load_data_proveedores import cargar_no_localizados
from pemex_contratos.load_data_proveedores import cargar_padron_proveedores
from pemex_contratos.load_data_proveedores import cargar_lista_contribuyentes_69b
from pemex_contratos.load_data_proveedores import cargar_particulares_sancionados
from pemex_contratos.load_data_proveedores import cargar_proveedores_sancionados

pesos_featues = {
    "no_en_padron_proveedores": 1,
    "es_empresa_fantasma": 3,
    "empresa_no_localizada": 1,
    "tasa_exito_alta": 1,
    "empresa_reciente": 3,
    "market_share_contratos_riesgoso": 1,
    "market_share_monto_riesgoso": 1,
    "empresa_sancionada": 3,
    "no_encontrado_en_siscep": 2,
    "discrepancia_inai_siscep": 1,
    "solo_una_cotizacion": 2,
    "falta_asistencia_junta": 1,
    "solo_una_oferta": 2,
    "participacion_baja": 1,
    "periodo_corto_fecha_junta_fecha_convocatoria": 2,
    "periodo_corto_fecha_contrato_fecha_junta": 2,
    "tuvo_convenios_modificatorios": 1,
    "plazo_corto_entrega": 2,
    "misma_competencia_y_tasa_exito_alta": 3,
}


if __name__ == "__main__":
    # Entradas

    n_contratos = 2029
    n_expedientes = 1421
    n_proveedores = 1089
    # rutas contrataciones
    base_path_inai = "../data/raw/inai/"
    path_siscep = "../data/processed/contrataciones_pemex_siscep.csv"
    # ruatas informacion proveedores
    path_no_localizados = "../data/raw/No localizados.csv"
    path_padron = "../data/raw/padron_proveedores/"
    path_particulares_sancionados = "../data/raw/s3-particulares-sfp.json"
    path_proveedores_sancionados = "../data/raw/proveedores_sancionados.csv"
    path_listado = "../data/raw/Listado_Completo_69-B.csv"
    # ruta final de los features y el score
    path_output = "../data/features/features_binarios_nivel_contrato.csv"

    # Carga y pipeline de datos

    # contratos del portal de transparencia
    df_inai = cargar_todos_procedimientos(base_path_inai)
    # contratos de la página del siscep
    df_siscep = pd.read_csv(
        path_siscep,
        dtype={"montos_maximos_mxn": float, "montos_minimos_mxn": float},
        parse_dates=[
            "publicado",
            "fecha_fallo",
            "fecha_recepcion_propuestas",
            "fecha_preguntas_y_respuestas",
            "fecha_creacion_empresa_rfc",
        ],
    )
    # tablas sobre proveedores y contratistas
    no_localizados = cargar_no_localizados(path_no_localizados)
    padron_proveedores = cargar_padron_proveedores(path_padron)
    particulares_sancionados = cargar_particulares_sancionados(
        path_particulares_sancionados
    )
    proveedores_sancionados = cargar_proveedores_sancionados(
        path_proveedores_sancionados
    )
    fantasma = cargar_lista_contribuyentes_69b(path_listado)

    # Tablas con informacion adicional sobre los contratos
    df_ofertas = cargar_tabla_ofertas(base_path_inai)
    df_cotizaciones = cargar_tabla_cotizaciones(base_path_inai)
    df_asistentes = cargar_tabla_asistentes_junta_aclaraciones(base_path_inai)
    df_contratantes = cargar_tabla_posibles_contratantes(base_path_inai)

    # Generar features
    df_contrato_binarios = features_binarios_contratos(df_inai, n_contratos)
    df_procedimiento_binarios = features_binarios_procedimientos(
        df_inai,
        df_siscep,
        ofertas=df_ofertas,
        cotizaciones=df_cotizaciones,
        posibles_contratantes=df_contratantes,
        asistentes_junta=df_asistentes,
        n_procedimientos=n_expedientes,
    )
    df_proveedor_binarios = features_binarios_proveedores(
        df_inai,
        ofertas=df_ofertas,
        proveedores_fantasma=fantasma,
        proveedores_no_localizados=no_localizados,
        proveedores_sancionados=proveedores_sancionados,
        particulares_sancionados=particulares_sancionados,
        padron_proveedores=padron_proveedores,
        n_proveedores=n_proveedores,
    )
    # Se unen los features a nivel razon_social, num_evento y numero_contrato
    cols = ["num_evento", "numero_contrato", "razon_social_simple"]
    features_binarios = df_inai.groupby(cols).monto.sum().reset_index()
    features_binarios = pd.merge(
        features_binarios, df_proveedor_binarios, "left", "razon_social_simple"
    )
    features_binarios = pd.merge(
        features_binarios, df_procedimiento_binarios, "left", "num_evento"
    )
    features_binarios = pd.merge(
        features_binarios,
        df_contrato_binarios,
        "left",
        ["numero_contrato", "num_evento"],
    )
    assert features_binarios.shape == (2117, 23)
    # se agrupan los features de razon social y se deja la tabla a nivel contrato
    df_nivel_contrato = features_binarios.groupby(
        ["num_evento", "numero_contrato"]
    ).sum()
    assert df_nivel_contrato.shape == (2029, 20)
    features_cols = [
        c
        for c in df_nivel_contrato.columns
        if c not in {"monto", "num_evento", "numero_contrato"}
    ]
    # Con la agrupación de empresas los features con valores mayores a cero se de
    # marcan como uno
    for c in features_cols:
        df_nivel_contrato.loc[:, c] = np.where(df_nivel_contrato[c] > 0, 1, 0)

    df_nivel_contrato = df_nivel_contrato.reset_index()
    pesos = pd.Series(pesos_featues).reindex(features_cols).to_numpy().reshape((19, 1))
    matriz_features = df_nivel_contrato.loc[:, features_cols].fillna(0).to_numpy()

    df_nivel_contrato = df_nivel_contrato.assign(
        sum_of_features=df_nivel_contrato[features_cols].fillna(0).sum(axis=1),
        weighted_sum_of_features=pd.Series((matriz_features @ pesos).flatten()),
    )
    df_nivel_contrato = df_nivel_contrato.assign(
        log_monto_times_weighted_sum=(
            np.log(df_nivel_contrato.monto) * df_nivel_contrato.weighted_sum_of_features
        )
    )
    assert df_nivel_contrato.shape == (2029, 25)
    df_nivel_contrato.to_csv(path_output, index=False, quoting=1, encoding="utf-8")
    print("Ejecución terminada.")
