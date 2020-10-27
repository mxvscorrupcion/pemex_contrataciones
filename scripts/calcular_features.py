"""Script que genera una tabla con los valores de los features
a nivel contrato, procedimiento y proveedor. También contiene los features
binarios"""
import pandas as pd
from pemex_contratos.inai.load_data import cargar_todos_procedimientos
from pemex_contratos.inai.load_data import cargar_tabla_ofertas
from pemex_contratos.inai.load_data import cargar_tabla_cotizaciones
from pemex_contratos.inai.load_data import cargar_tabla_asistentes_junta_aclaraciones
from pemex_contratos.inai.load_data import cargar_tabla_posibles_contratantes
from pemex_contratos.load_data_proveedores import cargar_no_localizados
from pemex_contratos.load_data_proveedores import cargar_padron_proveedores
from pemex_contratos.load_data_proveedores import cargar_lista_contribuyentes_69b
from pemex_contratos.load_data_proveedores import cargar_particulares_sancionados
from pemex_contratos.load_data_proveedores import cargar_proveedores_sancionados
from pemex_contratos.inai.features_contrato import (
    diferencia_fecha_contrato_fecha_junta,
    diferencia_fecha_junta_fecha_convocatoria,
    diferencia_fecha_termino_e_inicio_plazo,
    con_convenios_modificatorios,
)
from pemex_contratos.inai.features_procedimiento import (
    adjudicaciones_con_una_cotizacion,
    discrepancia_entre_inai_y_siscep,
    expediente_en_inai_no_existe_en_siscep,
    falta_asistencia_junta_aclaraciones,
    ofertas_unicas_en_licitaciones_e_invitaciones,
    participacion_baja_en_expediente,
)
from pemex_contratos.inai.features_proveedor import (
    empresa_creada_recientemente,
    empresa_no_en_padron_proveedores,
    empresa_no_localizada_sat,
    market_share_por_contratos,
    market_share_por_monto,
    proveedores_y_particulares_sancionados,
    reportada_como_empresa_fantasma,
    tasa_exito_proveedor,
    participacion_conjunta_sospechosa,
)


if __name__ == "__main__":
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
    path_output = "../data/features/informacion_features.csv"
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

    # Features a nivel contrato
    dfs = [
        diferencia_fecha_junta_fecha_convocatoria(df_inai),
        diferencia_fecha_contrato_fecha_junta(df_inai),
        con_convenios_modificatorios(df_inai),
        diferencia_fecha_termino_e_inicio_plazo(df_inai),
    ]
    dfs = [df.set_index(["num_evento", "numero_contrato"]) for df in dfs]
    assert all([df.shape[0] == n_contratos for df in dfs])
    df_features_contratos = pd.concat(dfs, axis=1, ignore_index=False).reset_index()

    # features nivel expediente
    dfs = [
        expediente_en_inai_no_existe_en_siscep(df_inai, df_siscep),
        discrepancia_entre_inai_y_siscep(df_inai, df_siscep),
        adjudicaciones_con_una_cotizacion(df_inai, df_cotizaciones),
        falta_asistencia_junta_aclaraciones(df_inai, df_asistentes),
        ofertas_unicas_en_licitaciones_e_invitaciones(df_inai, df_ofertas),
        participacion_baja_en_expediente(df_inai, df_contratantes),
    ]
    dfs = [df.set_index("num_evento") for df in dfs]
    assert all([df.shape[0] == n_expedientes for df in dfs])
    df_features_expedientes = (
        pd.concat(dfs, axis=1, ignore_index=False)
        .reset_index()
        .rename(columns={"index": "num_evento"})
    )

    # features nivel proveedor
    dfs = [
        empresa_no_en_padron_proveedores(df_inai, padron_proveedores),
        reportada_como_empresa_fantasma(df_inai, fantasma),
        empresa_no_localizada_sat(df_inai, no_localizados),
        tasa_exito_proveedor(df_inai, df_ofertas, 0.5),
        empresa_creada_recientemente(df_inai),
        market_share_por_contratos(df_inai),
        market_share_por_monto(df_inai),
        participacion_conjunta_sospechosa(df_inai, df_ofertas, 5, 0.5, 0.5),
        proveedores_y_particulares_sancionados(
            df_inai, proveedores_sancionados, particulares_sancionados
        ),
    ]
    dfs = [df.set_index("razon_social_simple") for df in dfs]
    assert all([df.shape[0] == n_proveedores for df in dfs])
    df_features_empresas = (
        pd.concat(dfs, axis=1, ignore_index=False)
        .reset_index()
        .rename(columns={"index": "razon_social_simple"})
    )

    # unir los features a nivel razon_social, num_evento y numero_contrato
    features_all = (
        df_inai.groupby(["num_evento", "numero_contrato", "razon_social_simple"])
        .monto.sum()
        .reset_index()
    )
    assert features_all.shape == (2117, 4)
    features_all = pd.merge(
        features_all, df_features_empresas, "left", "razon_social_simple"
    )
    features_all = pd.merge(features_all, df_features_expedientes, "left", "num_evento")
    features_all = pd.merge(
        features_all, df_features_contratos, "left", ["numero_contrato", "num_evento"]
    )
    features_all.to_csv(path_output, index=False, quoting=1, encoding="utf-8")
