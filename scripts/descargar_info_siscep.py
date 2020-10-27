import os
import time
import random
import itertools
from pathlib import Path
import helium
import gazpacho
import requests
import joblib
import pandas as pd
from typing import List


helium.Config.implicit_wait_secs = 30

user_agent_list = [
   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:75.0) Gecko/20100101 Firefox/75.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/80.0.3987.163 Chrome/80.0.3987.163 Safari/537.36',
    #Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
]


def table_in_page(html: str) -> bool:
    """Revisa que el documento tenga la tabla que se quiere extraer"""
    page_soup = gazpacho.Soup(html)
    table_soup = page_soup.find('table', {'id': 'mitabla'})
    if table_soup:
        return True
    return False


# TODO: ver que tiene los campos indicados
def get_table_in_page(html: str) -> pd.DataFrame:
    page_soup = gazpacho.Soup(html)
    table_soup = page_soup.find('table', {'id': 'mitabla'})
    # Extraer data-itemid de los botones
    rows = table_soup.find('tr')
    # drop header
    rows.pop(0)
    last_column = [row.find('td')[-1] for row in rows]
    buttons = [row.find('input') for row in last_column]
    ids = [int(b.attrs.get('data-itemid')) for b in buttons]
    df = pd.read_html(str(table_soup))[0]
    df = df.assign(data_itemid=ids)
    return df


def save_file(
        df_sub: pd.DataFrame,
        base_path: str,
        data_itemid: int,
        chunk_size: int = 1024
):
    data_itemid_path = os.path.join(base_path, f'{data_itemid}')
    os.makedirs(data_itemid_path)
    links = [link for link in df_sub.file_link]
    session = requests.Session()
    headers = requests.utils.default_headers()
    user_agent = random.choice(user_agent_list)
    headers['User-Agent'] = user_agent
    for file_link in links:
        file_name = file_link.split('/')[-1]
        save_path = os.path.join(data_itemid_path, file_name)
        # https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
        r = session.get(file_link, headers=headers, stream=True, timeout=60)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        sleep_secs = random.randint(1, 5)
        time.sleep(sleep_secs)


def download_scraped_tables(url: str, path: str):
    tables: List[pd.DataFrame] = []
    # data_itemid_to_links: Dict[int, List[str]] = {}
    data_itemid_and_links = []
    driver = helium.start_firefox(url, headless=True)
    # si la tabla existe
    if table_in_page(driver.page_source):
        # Recorremos todas las paginas
        while True:
            # guardamos las tablas de cada página
            tables.append(get_table_in_page(driver.page_source))
            # buscamos los botones y los aplatamos para que
            # la página muestre los links de descarga
            buttons = helium.find_all(helium.Button('Ver documentos'))
            for button in buttons:
                helium.click(button)
                close_x = helium.find_all(helium.S('.close'))[0]
                helium.click(close_x)
            # Después de aplastar los botones tenemos los webuis activados
            soup = gazpacho.Soup(driver.page_source)
            popups = soup.find('div', {'class': 'webui-popover-content'})
            if isinstance(popups, gazpacho.Soup):
                popups = [popups]
            # Vamos a guardar todos los links de los archivos
            for web_ui in popups:
                paragraphs = web_ui.find('p')
                links = []
                for p in paragraphs:
                    # find the links
                    file_link = p.find('a')
                    if file_link:
                        href = file_link.attrs.get('href')
                        links.append(href)
                if len(links):
                    # extar el data-itemid
                    data_itemid = int(links[0].split('/')[-2])
                    # agregamos el dominio principal a los links
                    # TODO: podemos ponerlo como parametro
                    for link in links:
                        record = {
                            'data_itemid': data_itemid,
                            'file_link': f'https://www.pemex.com/{link}'
                        }
                        data_itemid_and_links.append(record)
            # encontrar el boton > (next)
            next_buttons = helium.find_all(helium.S('.next'))
            # si el botón ya no aparece terminamos
            if len(next_buttons) == 0:
                break
            # Presionar el boton > (next)
            helium.click(next_buttons[0])
        helium.kill_browser()
        eventos_path = os.path.join(path, 'eventos.csv')
        links_path = os.path.join(path, 'links.csv')
        df_eventos = pd.concat(tables, axis=0, ignore_index=True, sort=False)
        df_links = pd.DataFrame(data_itemid_and_links)
        # descargar tablas
        df_eventos.to_csv(eventos_path, index=False, quoting=1, encoding='utf-8')
        df_links.to_csv(links_path, index=False, quoting=1, encoding='utf-8')
    else:
        m = f'La página {url} \n no contiene una tabla con documentos'
        print(m)
        helium.kill_browser()


def main(base_url: str, procedimiento: str, negocio: str, base_path: str):
    url = f'{base_url}/{procedimiento}/Paginas/{negocio}.aspx'
    path = Path(os.path.join(base_path, tipo_proc, negocio))
    if not path.exists():
        os.makedirs(path.as_posix())
    eventos_path = path / 'eventos.csv'
    links_path = path / 'links.csv'
    # si los archivos no existen se scrapean
    if not eventos_path.exists() or not links_path.exists():
        print('eventos.csv o links.csv no existen. El scraping comienza ...')
        download_scraped_tables(url, path.as_posix())
        print('... scraping terminado')
    # si existen se procede a descargar archivos
    if eventos_path.exists() and links_path.exists():
        df_eventos = pd.read_csv(eventos_path)
        df_links = pd.read_csv(links_path, dtype={'data_itemid': int})
        folders = [int(p.name) for p in path.iterdir() if p.is_dir()]
        # no se van a pedir los archivos ya descargados
        df_links = df_links.loc[~df_links.data_itemid.isin(folders)]
        print(df_links.shape, df_links.data_itemid.nunique())
        # TODO: si ya se pidieron todos salir
        if df_links.shape[0] == 0:
            return df_eventos, df_links
        df_grouped_links = df_links.groupby('data_itemid')
        # descargar archivos
        joblib.Parallel(n_jobs=-1, prefer="threads", verbose=11)(
            joblib.delayed(save_file)(df_sub, path.as_posix(), data_itemid)
            for data_itemid, df_sub in df_grouped_links
        )
        return df_eventos, df_links
    return None, None


if __name__ == '__main__':
    base_url = 'https://www.pemex.com/procura/procedimientos-de-contratacion'
    base_output_path = '../data/raw/documentos_siscep'
    # tipos de procedimientos
    procedimientos = ('adjudicaciones', 'concursosabiertos', 'invitaciones')
    negocios = (
        'Pemex-Logística',
        'Pemex-Fertilizantes',
        'Pemex-Etileno',
        'Pemex',
        'Pemex-Exploración-y-Producción',
        'Pemex-Cogeneración-y-Servicios',
        'Pemex-Transformación-Industrial',
        'Pemex-Perforación-y-Servicios',
    )
    # Producto cartesiano procedimientos x negocios
    pares = itertools.product(procedimientos, negocios)
    # se corrige un caso que no sigue el patron del producto cartesiano
    # la url es diferente
    pares = [
        (proc, 'Petroleos-Mexicanos')
        if (proc == 'invitaciones' and negocio == 'Pemex') else (proc, negocio)
        for proc, negocio in pares
    ]
    # TODO: se puede mejorar el proceso si primero solo se descargan los
    #  links y nombres de archivos y en otro programa se hacen las peticiones
    for tipo_proc, negocio in pares:
        print(f'Empezando con {tipo_proc}, {negocio} ...')
        df_eventos_aux, df_links_aux = main(base_url, tipo_proc, negocio, base_output_path)
        print(f'... Terminando con {tipo_proc}, {negocio}')
        print('-' * 100)
