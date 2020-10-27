"""Funciones para calcular los features a nivel contrato"""
import pandas as pd


def features_binarios_contratos(
        procedimientos: pd.DataFrame,
        n_contratos: int = 2029) -> pd.DataFrame:
    # esta funcion sólo regresa los features binarios de la tabla final
    dfs = [
        diferencia_fecha_junta_fecha_convocatoria(procedimientos),
        diferencia_fecha_contrato_fecha_junta(procedimientos),
        con_convenios_modificatorios(procedimientos),
        diferencia_fecha_termino_e_inicio_plazo(procedimientos),
    ]
    dfs = [df.set_index(['num_evento', 'numero_contrato']) for df in dfs]
    if not all(df.shape[0] == n_contratos for df in dfs):
        m = ('Existen shapes que no son iguales al número de contratos indicado. '
             f'Los shapes son los siguientes: {[df.shape[0] for df in dfs]}')
        raise ValueError(m)
    df_features_contratos = pd.concat(dfs, axis=1, ignore_index=False).reset_index()
    # se quitan los features que no son binarios
    cols = [
        'diferenica_plazo_entrega',
        'diferencia_fecha_contrato_fecha_junta',
        'diferencia_fecha_junta_fecha_convocatoria',
    ]
    df_features_contratos = df_features_contratos.drop(cols, axis=1)
    return df_features_contratos


def diferencia_fecha_junta_fecha_convocatoria(
        procedimientos: pd.DataFrame,
        cuantil_max: float = 0.1
) -> pd.DataFrame:
    # feature 3
    cols = [
        'ID', 'num_evento', 'materia', 'numero_contrato',
        'fecha_junta_aclaraciones', 'fecha_convocatoria'
    ]
    df = procedimientos.loc[:, cols]
    dif = (df.fecha_junta_aclaraciones - df.fecha_convocatoria).dt.days
    df = df.assign(diferencia_fecha_junta_fecha_convocatoria=dif)
    # se agrupan los consorcios
    df = (df.groupby(['num_evento', 'ID', ], as_index=False)
          .agg({'diferencia_fecha_junta_fecha_convocatoria': 'min',
                'materia': 'unique', 'numero_contrato': 'unique'}))
    materia = (df.materia.map(lambda m: m[0] if m else m)
               .replace('Dos Bocas', 'Obra pública'))
    df = df.assign(
        materia=materia,
        numero_contrato=df.numero_contrato.map(lambda c: c[0] if c else c),
    )
    dfs = []
    for g, df_group in df.groupby('materia'):
        diferencia = df_group.diferencia_fecha_junta_fecha_convocatoria
        cuantil = diferencia.dropna().quantile(cuantil_max)
        if not pd.isna(cuantil):
            cond = (diferencia < cuantil).astype(int)
            df_group = df_group.assign(
                periodo_corto_fecha_junta_fecha_convocatoria=cond
            )
        dfs.append(df_group)
    feature = pd.concat(dfs, axis=0, ignore_index=True)
    final_cols = [
        'num_evento', 'numero_contrato',
        'diferencia_fecha_junta_fecha_convocatoria',
        'periodo_corto_fecha_junta_fecha_convocatoria'
    ]
    feature = feature.loc[:, final_cols]
    return feature


def diferencia_fecha_contrato_fecha_junta(
        procedimientos: pd.DataFrame,
        cuantil_max: float = 0.1
) -> pd.DataFrame:
    # TODO: tener cuidado al usar numero de contrato porque se va Dos Bocas
    # feature 4
    cols = [
        'ID', 'num_evento', 'numero_contrato', 'materia',
        'fecha_junta_aclaraciones', 'fecha_del_contrato'
    ]
    df = procedimientos.loc[:, cols]
    dif = (df.fecha_del_contrato - df.fecha_junta_aclaraciones).dt.days
    df = df.assign(diferencia_fecha_contrato_fecha_junta=dif)
    df = (df.groupby(['num_evento', 'ID', ], as_index=False)
          .agg({'diferencia_fecha_contrato_fecha_junta': 'min',
                'materia': 'unique', 'numero_contrato': 'unique'}))
    materia = (df.materia.map(lambda m: m[0] if m else m)
               .replace('Dos Bocas', 'Obra pública'))
    df = df.assign(
        materia=materia,
        numero_contrato=df.numero_contrato.map(lambda c: c[0] if c else c),
    )
    dfs = []
    for g, df_group in df.groupby('materia'):
        diferencia = df_group.diferencia_fecha_contrato_fecha_junta
        cuantil = diferencia.dropna().quantile(cuantil_max)
        if not pd.isna(cuantil):
            cond = (diferencia < cuantil).astype(int)
            df_group = df_group.assign(
                periodo_corto_fecha_contrato_fecha_junta=cond
            )
        dfs.append(df_group)
    feature = pd.concat(dfs, axis=0, ignore_index=True)
    final_cols = [
        'num_evento', 'numero_contrato',
        'diferencia_fecha_contrato_fecha_junta',
        'periodo_corto_fecha_contrato_fecha_junta'
    ]
    feature = feature.loc[:, final_cols]
    return feature


def con_convenios_modificatorios(procedimientos: pd.DataFrame) -> pd.DataFrame:
    # feature 14
    cols = ['num_evento', 'numero_contrato', 'tuvo_convenios_modificatorios']
    contratos = procedimientos.loc[:, cols].drop_duplicates()
    mapeo = {'No': 0, 'Si': 1}
    tuvo_convenios = contratos.tuvo_convenios_modificatorios.map(mapeo)
    contratos = contratos.assign(tuvo_convenios_modificatorios=tuvo_convenios)
    return contratos


def diferencia_fecha_termino_e_inicio_plazo(
        procedimientos: pd.DataFrame, cuantil_max: float = 0.1) -> pd.DataFrame:
    # feature 24
    cols = [
        'ID', 'num_evento', 'numero_contrato', 'materia',
        'fecha_inicio_plazo_entrega', 'fecha_termino_plazo_entrega'
    ]
    df = procedimientos.loc[:, cols]
    dif = (df.fecha_termino_plazo_entrega - df.fecha_inicio_plazo_entrega).dt.days
    df = df.assign(diferenica_plazo_entrega=dif)
    # se agrupan los consorcios
    df = (df.groupby(['num_evento', 'ID', ], as_index=False)
          .agg({'diferenica_plazo_entrega': 'min',
                'materia': 'unique', 'numero_contrato': 'unique'}))
    materia = (df.materia.map(lambda m: m[0] if m else m)
               .replace('Dos Bocas', 'Obra pública'))
    df = df.assign(
        materia=materia,
        numero_contrato=df.numero_contrato.map(lambda c: c[0] if c else c),
    )
    dfs = []
    for g, df_group in df.groupby('materia'):
        diferencia = df_group.diferenica_plazo_entrega
        cuantil = diferencia.dropna().quantile(cuantil_max)
        if not pd.isna(cuantil):
            cond = (diferencia < cuantil).astype(int)
            df_group = df_group.assign(
                plazo_corto_entrega=cond
            )
        dfs.append(df_group)
    feature = pd.concat(dfs, axis=0, ignore_index=True)
    final_cols = [
        'num_evento', 'numero_contrato',
        'diferenica_plazo_entrega',
        'plazo_corto_entrega'
    ]
    feature = feature.loc[:, final_cols]
    return feature

