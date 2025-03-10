import geopandas as gpd
import pandas as pd
import ee
import geemap
import logging
import time
from multiprocessing import Pool
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

class Sentinel2IndexCalculator:
    """
    A class for calculating various vegetation indices from Sentinel-2 satellite imagery.

    Attributes:
        CLOUD_VALUES (list): List of cloud values used for cloud masking.
        BANDS_NDVI (list): List of bands used for calculating NDVI.
        BANDS_EVI (dict): Dictionary mapping bands used for calculating EVI.
        EVI_COEFFICIENTS (dict): Dictionary of coefficients used for calculating EVI.
        SCALE (int): Scale used for reducing the image resolution.
        DATE_FORMAT (str): Date format used for formatting dates.
        COLLECTION (str): Collection name for Sentinel-2 satellite imagery.
        L_SAVI (float): L parameter used for calculating SAVI.
        SLOPE_PVI (int): Slope parameter used for calculating PVI.
        INTERCEPT_PVI (int): Intercept parameter used for calculating PVI.
        SLOPE_WSI (int): Slope parameter used for calculating WSI.
        INTERCEPT_WSI (int): Intercept parameter used for calculating WSI.

    Methods:
        __init__(start_date, end_date, region, id_column='id', batch_size=50, max_workers=5, timeout=300): Initializes the Sentinel2IndexCalculator object.
        mask_clouds(image): Masks clouds in the input image.
        filter_sentinel_collection(): Filters the Sentinel-2 image collection based on date and region.
        calculate_ndvi(image): Calculates the NDVI index for the input image.
        calculate_evi(image): Calculates the EVI index for the input image.
        calculate_lai(image): Calculates the LAI index for the input image.
        calculate_savi(image): Calculates the SAVI index for the input image.
        calculate_pvi(image): Calculates the PVI index for the input image.
        calculate_wsi(image): Calculates the WSI index for the input image.
        add_date_to_image(image): Adds the date property to the input image.
        calculate_mean_index(image, region): Calculates the mean index for the input image and region.
        get_mean_index_collection(image_collection, calculate_index): Gets the mean index collection for the input image collection and index calculation function.
        get_results_as_dataframe(feature_collections, index_names): Converts the feature collections to a pandas DataFrame.
        compute_batch_indices(b): Computes the indices for a batch of images.
        _process_batch_with_retry(b, max_retries=3, retry_delay=5): Processes a batch with retry mechanism in case of failure.
        aggregate_all_batches(buffer_size=5): Aggregates all batches of images and returns the results as a DataFrame.
        flatten_feature_collections(feature_collections): Flattens the feature collections.
    """
    VEGETATION_VALUE = 4
    BANDS_NDVI = ['B8', 'B4']
    BANDS_EVI = {'nir': 'B8', 'red': 'B4', 'blue': 'B2'}
    EVI_COEFFICIENTS = {'G': 2.5, 'C1': 6, 'C2': 7.5, 'L': 1}
    SCALE = 10
    DATE_FORMAT = '%d-%m-%Y'
    COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
    L_SAVI = 0.5
    SLOPE_PVI = 1
    INTERCEPT_PVI = 0
    SLOPE_WSI = 1
    INTERCEPT_WSI = 0

    def __init__(self, start_date, end_date, region, id_column='id', batch_size=50, max_workers=5, timeout=300):
        """
        Inicializa o objeto Sentinel2IndexCalculator.

        Args:
            start_date (str): Data inicial da coleção de imagens no formato 'dd-mm-yyyy'.
            end_date (str): Data final da coleção de imagens no formato 'dd-mm-yyyy'.
            region (ee.Geometry): Região de interesse para filtrar a coleção de imagens.
            id_column (str, optional): Nome da coluna a ser usada como ID. Padrão é 'id'.
            batch_size (int, optional): Tamanho de cada lote para processamento da coleção de imagens. Padrão é 50.
            max_workers (int, optional): Número máximo de workers para processamento paralelo. Padrão é 5.
            timeout (int, optional): Tempo máximo em segundos para processar um batch. Padrão é 300.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.region = region
        self.id_column = id_column
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.sentinel2 = self.filter_sentinel_collection()
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def mask_clouds(image):
        """
        Masks out clouds in the given image using the Sentinel-2 cloud mask values.

        Args:
            image (ee.Image): The input image.

        Returns:
            ee.Image: The masked image with clouds removed.
        """
        mask = image.select('SCL').neq(Sentinel2IndexCalculator.VEGETATION_VALUE)
        return image.updateMask(mask)

    def filter_sentinel_collection(self):
        """
        Filters the Sentinel image collection based on region, start date, end date, and cloud masking.

        Returns:
            ee.ImageCollection: Filtered Sentinel image collection.
        """
        return (ee.ImageCollection(Sentinel2IndexCalculator.COLLECTION)
                .filterBounds(self.region)
                .filterDate(ee.Date(self.start_date), ee.Date(self.end_date))
                .map(self.mask_clouds))

    @staticmethod
    def calculate_ndvi(image):
        """
        Calculates the Normalized Difference Vegetation Index (NDVI) for an image.
        NDVI is calculated from the normalized difference between the Near Infrared (NIR) and Red bands.
        """
        return image.normalizedDifference(Sentinel2IndexCalculator.BANDS_NDVI)
    
    @staticmethod
    def calculate_evi(image):
        """
        Calculates the Enhanced Vegetation Index (EVI) for an image.
        EVI is a more optimized vegetation index that corrects for some distortions in the reflected light and atmosphere.
        """
        nir = image.select(Sentinel2IndexCalculator.BANDS_EVI['nir'])
        red = image.select(Sentinel2IndexCalculator.BANDS_EVI['red'])
        blue = image.select(Sentinel2IndexCalculator.BANDS_EVI['blue'])
        G = Sentinel2IndexCalculator.EVI_COEFFICIENTS['G']
        C1 = Sentinel2IndexCalculator.EVI_COEFFICIENTS['C1']
        C2 = Sentinel2IndexCalculator.EVI_COEFFICIENTS['C2']
        L = Sentinel2IndexCalculator.EVI_COEFFICIENTS['L']
        evi = nir.subtract(red).divide(nir.add(red.multiply(C1).subtract(blue.multiply(C2)).add(L))).multiply(G)
        return evi
    
    @staticmethod
    def calculate_lai(image):
        """
        Calculates the Leaf Area Index (LAI) for an image.
        LAI is a dimensionless quantity that characterizes plant canopies. It is defined as the one-sided green leaf area per unit ground area.
        """
        nir = image.select('B8')
        red = image.select('B4')
        lai = nir.subtract(red)
        return lai

    @staticmethod
    def calculate_savi(image):
        """
        Calculates the Soil Adjusted Vegetation Index (SAVI) for an image.
        SAVI is a vegetation index that attempts to minimize soil brightness influences using a soil-brightness correction factor.
        """
        nir = image.select('B8')
        red = image.select('B4')
        savi = nir.subtract(red).multiply(1 + Sentinel2IndexCalculator.L_SAVI).divide(nir.add(red).add(Sentinel2IndexCalculator.L_SAVI))
        return savi

    @staticmethod
    def calculate_pvi(image):
        """
        Calculates the Perpendicular Vegetation Index (PVI) for an image.
        PVI is a vegetation index that emphasizes perpendicularity to reduce soil noise.
        """
        nir = image.select('B8')
        red = image.select('B4')
        pvi = nir.subtract(red.multiply(Sentinel2IndexCalculator.SLOPE_PVI)).subtract(Sentinel2IndexCalculator.INTERCEPT_PVI)
        return pvi

    @staticmethod
    def calculate_wsi(image):
        """
        Calculates the Water Stress Index (WSI) for an image.
        WSI is an index used to quantify water stress in plants.
        """
        nir = image.select('B8')
        swir = image.select('B12') 
        wsi = nir.subtract(swir.multiply(Sentinel2IndexCalculator.SLOPE_WSI)).subtract(Sentinel2IndexCalculator.INTERCEPT_WSI)
        return wsi

    @staticmethod
    def add_date_to_image(image):
        """
        Adds a 'date' property to an image.
        The 'date' property is formatted from the image's date.
        """
        return image.set('date', image.date().format())

    @staticmethod
    def calculate_mean_index(image, region):
        """
        Calculates the mean index of an image over a specified region.
        The mean index is calculated using a reducer over the specified region at a specified scale.
        """
        mean_index = image.reduceRegions(
            collection=region,
            reducer=ee.Reducer.mean(),
            scale=Sentinel2IndexCalculator.SCALE)
        return mean_index.set('date', image.get('date'))

    def get_mean_index_collection(self, image_collection, calculate_index):
        """
        Gets a collection of mean indices from an image collection.
        The mean indices are calculated using the specified index calculation method.
        """
        image_collection = image_collection.map(self.add_date_to_image)
        image_collection = image_collection.map(
            lambda image: self.calculate_mean_index(calculate_index(image), self.region)
            .map(lambda feature: feature.set('date', image.get('date'))))
        return image_collection

    def get_results_as_dataframe(self, feature_collections, index_names):
        """
        Converts the feature collections to a pandas DataFrame.

        Args:
            feature_collections (list): The list of feature collections.
            index_names (list): The list of index names.

        Returns:
            pd.DataFrame: The results as a pandas DataFrame.
        """
        results = [fc.getInfo() for fc in feature_collections]

        all_data = []
        for features in zip(*[r['features'] for r in results]):
            data = {self.id_column: features[0]['properties'].get(self.id_column),
                    'date': features[0]['properties'].get('date')}
            for i, feature in enumerate(features):
                data[index_names[i]] = feature['properties'].get('mean')
            all_data.append(data)

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date']).dt.strftime(Sentinel2IndexCalculator.DATE_FORMAT)
        return df
    
    def compute_batch_indices(self, b):
        """
        Computa os índices para um lote de imagens.

        Args:
            b (int): O número do lote.

        Returns:
            pd.DataFrame: Os resultados como um DataFrame do pandas.
            
        Raises:
            Exception: Se ocorrer um erro durante o processamento do lote.
        """
        try:
            start = b * self.batch_size
            end = start + self.batch_size
            batch = self.sentinel2.toList(self.batch_size, start)
            batch_image_collection = ee.ImageCollection(batch)

            # Etapas de processamento para índices
            calculate_indices = [self.calculate_ndvi, self.calculate_evi, self.calculate_lai, 
                                self.calculate_savi, self.calculate_pvi, self.calculate_wsi]
            index_names = ['ndvi', 'evi', 'lai', 'savi', 'pvi', 'wsi']

            # Processa cada índice e achata as coleções
            mean_index_collections = [self.get_mean_index_collection(batch_image_collection, index_calc) 
                                    for index_calc in calculate_indices]
            mean_index_fcs = [self.flatten_feature_collections(index_collection) 
                            for index_collection in mean_index_collections]

            # Constrói o dataframe a partir dos resultados
            df = self.get_results_as_dataframe(mean_index_fcs, index_names)
            
            self.logger.info(f"Lote {b} processado com sucesso: {len(df)} registros")
            return df
        except Exception as e:
            self.logger.error(f"Erro ao processar lote {b}: {str(e)}")
            raise

    def _process_batch_with_retry(self, b, max_retries=3, retry_delay=5):
        """
        Processa um lote com sistema de tentativas em caso de falha.
        
        Args:
            b (int): Número do lote
            max_retries (int): Número máximo de tentativas
            retry_delay (int): Tempo de espera entre tentativas em segundos
            
        Returns:
            pd.DataFrame or None: DataFrame com resultados ou None se falhar
        """
        for attempt in range(max_retries):
            try:
                return self.compute_batch_indices(b)
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Tentativa {attempt+1} falhou para lote {b}. Tentando novamente em {retry_delay}s")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Falha ao processar lote {b} após {max_retries} tentativas: {str(e)}")
                    return pd.DataFrame()  # Retorna DataFrame vazio em caso de falha final

    def aggregate_all_batches(self, buffer_size=5):
        """
        Agrega todos os lotes de imagens e retorna os resultados como DataFrame.
        
        Args:
            buffer_size (int): Tamanho do buffer para armazenar resultados antes de concatenar
                               Use valores menores para conjuntos grandes de dados
        
        Returns:
            pd.DataFrame: Os resultados agregados como DataFrame do pandas.
        """
        collection_size = self.sentinel2.size().getInfo()
        batches = collection_size // self.batch_size + (collection_size % self.batch_size > 0)
        
        self.logger.info(f"Iniciando processamento de {batches} lotes (total de {collection_size} imagens)")

        all_dataframes = []
        buffer = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submissão de todas as tarefas de processamento em lote como tarefas futuras
            futures = {executor.submit(self._process_batch_with_retry, b): b for b in range(batches)}
            
            # Coleta de resultados com barra de progresso
            with tqdm(total=batches, desc="Processando lotes") as progress:
                for future in as_completed(futures):
                    batch_num = futures[future]
                    try:
                        result_df = future.result(timeout=self.timeout)
                        buffer.append(result_df)
                        
                        # Concatena e limpa o buffer quando chegar ao tamanho máximo
                        if len(buffer) >= buffer_size:
                            if buffer:
                                all_dataframes.append(pd.concat(buffer, ignore_index=True))
                            buffer = []
                            
                    except TimeoutError:
                        self.logger.error(f"Timeout ao processar lote {batch_num}")
                    except Exception as e:
                        self.logger.error(f"Erro em lote {batch_num}: {str(e)}")
                    finally:
                        progress.update(1)
        
        # Adiciona quaisquer resultados restantes no buffer
        if buffer:
            all_dataframes.append(pd.concat(buffer, ignore_index=True))
            
        if not all_dataframes:
            self.logger.warning("Nenhum dado processado com sucesso!")
            return pd.DataFrame()
            
        final_df = pd.concat(all_dataframes, ignore_index=True)
        self.logger.info(f"Processamento concluído. Obtidos {len(final_df)} registros no total.")
        return final_df

    def flatten_feature_collections(self, feature_collections):
        """
        Achata as coleções de feições.

        Args:
            feature_collections (ee.FeatureCollection): As coleções de feições.

        Returns:
            ee.FeatureCollection: As coleções de feições achatadas.
        """
        return feature_collections.map(lambda feature: feature.set('date', feature.get('date'))).flatten()