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
        Initializes the Sentinel2IndexCalculator object.

        Args:
            start_date (str): Start date of the image collection in 'dd-mm-yyyy' format.
            end_date (str): End date of the image collection in 'dd-mm-yyyy' format.
            region (ee.Geometry): Region of interest to filter the image collection.
            id_column (str, optional): Name of the column to be used as ID. Default is 'id'.
            batch_size (int, optional): Size of each batch for processing the image collection. Default is 50.
            max_workers (int, optional): Maximum number of workers for parallel processing. Default is 5.
            timeout (int, optional): Maximum time in seconds to process a batch. Default is 300.
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
        Computes indices for a batch of images.

        Args:
            b (int): The batch number.

        Returns:
            pd.DataFrame: The results as a pandas DataFrame.
            
        Raises:
            Exception: If an error occurs during batch processing.
        """
        try:
            start = b * self.batch_size
            end = start + self.batch_size
            batch = self.sentinel2.toList(self.batch_size, start)
            batch_image_collection = ee.ImageCollection(batch)

            # Processing steps for indices
            calculate_indices = [self.calculate_ndvi, self.calculate_evi, self.calculate_lai, 
                                self.calculate_savi, self.calculate_pvi, self.calculate_wsi]
            index_names = ['ndvi', 'evi', 'lai', 'savi', 'pvi', 'wsi']

            # Process each index and flatten collections
            mean_index_collections = [self.get_mean_index_collection(batch_image_collection, index_calc) 
                                    for index_calc in calculate_indices]
            mean_index_fcs = [self.flatten_feature_collections(index_collection) 
                            for index_collection in mean_index_collections]

            # Build dataframe from results
            df = self.get_results_as_dataframe(mean_index_fcs, index_names)
            
            self.logger.info(f"Batch {b} processed successfully: {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error processing batch {b}: {str(e)}")
            raise

    def _process_batch_with_retry(self, b, max_retries=3, retry_delay=5):
        """
        Processes a batch with retry mechanism in case of failure.
        
        Args:
            b (int): Batch number
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Wait time between attempts in seconds
            
        Returns:
            pd.DataFrame or None: DataFrame with results or None if failed
        """
        for attempt in range(max_retries):
            try:
                return self.compute_batch_indices(b)
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt+1} failed for batch {b}. Retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to process batch {b} after {max_retries} attempts: {str(e)}")
                    return pd.DataFrame()  # Return empty DataFrame in case of final failure

    def aggregate_all_batches(self, buffer_size=5):
        """
        Aggregates all batches of images and returns the results as DataFrame.
        
        Args:
            buffer_size (int): Buffer size to store results before concatenation
                               Use smaller values for larger datasets
        
        Returns:
            pd.DataFrame: The aggregated results as a pandas DataFrame.
        """
        collection_size = self.sentinel2.size().getInfo()
        batches = collection_size // self.batch_size + (collection_size % self.batch_size > 0)
        
        self.logger.info(f"Starting processing of {batches} batches (total of {collection_size} images)")

        all_dataframes = []
        buffer = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks as future tasks
            futures = {executor.submit(self._process_batch_with_retry, b): b for b in range(batches)}
            
            # Collect results with progress bar
            with tqdm(total=batches, desc="Processing batches") as progress:
                for future in as_completed(futures):
                    batch_num = futures[future]
                    try:
                        result_df = future.result(timeout=self.timeout)
                        buffer.append(result_df)
                        
                        # Concatenate and clear buffer when reaching maximum size
                        if len(buffer) >= buffer_size:
                            if buffer:
                                all_dataframes.append(pd.concat(buffer, ignore_index=True))
                            buffer = []
                            
                    except TimeoutError:
                        self.logger.error(f"Timeout processing batch {batch_num}")
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_num}: {str(e)}")
                    finally:
                        progress.update(1)
        
        # Add any remaining results in buffer
        if buffer:
            all_dataframes.append(pd.concat(buffer, ignore_index=True))
            
        if not all_dataframes:
            self.logger.warning("No data processed successfully!")
            return pd.DataFrame()
            
        final_df = pd.concat(all_dataframes, ignore_index=True)
        self.logger.info(f"Processing completed. Obtained {len(final_df)} records in total.")
        return final_df

    def flatten_feature_collections(self, feature_collections):
        """
        Flattens the feature collections.

        Args:
            feature_collections (ee.FeatureCollection): The feature collections.

        Returns:
            ee.FeatureCollection: The flattened feature collections.
        """
        return feature_collections.map(lambda feature: feature.set('date', feature.get('date'))).flatten()

    def calculate_single_index(self, index_name, buffer_size=5):
        """
        Calculates a single specified index for all images.
        
        Args:
            index_name (str): Name of the index to calculate ('ndvi', 'evi', 'lai', 'savi', 'pvi', 'wsi')
            buffer_size (int): Buffer size to store results before concatenation
        
        Returns:
            pd.DataFrame: DataFrame containing only the data of the requested index
            
        Raises:
            ValueError: If the index name is not recognized
        """
        # Mapping index names to their calculation functions
        index_functions = {
            'ndvi': self.calculate_ndvi,
            'evi': self.calculate_evi,
            'lai': self.calculate_lai,
            'savi': self.calculate_savi,
            'pvi': self.calculate_pvi,
            'wsi': self.calculate_wsi
        }
        
        if index_name.lower() not in index_functions:
            raise ValueError(f"Index '{index_name}' not recognized. Available indices: {', '.join(index_functions.keys())}")
        
        calculate_index = index_functions[index_name.lower()]
        self.logger.info(f"Starting calculation of index {index_name.upper()}")
        
        collection_size = self.sentinel2.size().getInfo()
        batches = collection_size // self.batch_size + (collection_size % self.batch_size > 0)
        
        self.logger.info(f"Processing {batches} batches (total of {collection_size} images)")
        
        all_dataframes = []
        buffer = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for b in range(batches):
                futures[executor.submit(self._process_single_index_batch, b, calculate_index, index_name)] = b
                
            with tqdm(total=batches, desc=f"Processing {index_name.upper()}") as progress:
                for future in as_completed(futures):
                    batch_num = futures[future]
                    try:
                        result_df = future.result(timeout=self.timeout)
                        if not result_df.empty:
                            buffer.append(result_df)
                            
                            # Concatenate and clear buffer when reaching maximum size
                            if len(buffer) >= buffer_size:
                                all_dataframes.append(pd.concat(buffer, ignore_index=True))
                                buffer = []
                                
                    except TimeoutError:
                        self.logger.error(f"Timeout processing batch {batch_num}")
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_num}: {str(e)}")
                    finally:
                        progress.update(1)
        
        # Add any remaining results in buffer
        if buffer:
            all_dataframes.append(pd.concat(buffer, ignore_index=True))
            
        if not all_dataframes:
            self.logger.warning("No data processed successfully!")
            return pd.DataFrame()
            
        final_df = pd.concat(all_dataframes, ignore_index=True)
        self.logger.info(f"Processing of index {index_name.upper()} completed. Obtained {len(final_df)} records.")
        return final_df
    
    def _process_single_index_batch(self, b, calculate_index, index_name):
        """
        Processes a batch for a single specified index.
        
        Args:
            b (int): The batch number
            calculate_index (function): Function to calculate the index
            index_name (str): Name of the index being calculated
            
        Returns:
            pd.DataFrame: DataFrame containing the results for the batch
        """
        try:
            start = b * self.batch_size
            batch = self.sentinel2.toList(self.batch_size, start)
            batch_image_collection = ee.ImageCollection(batch)
            
            # Process only the requested index
            mean_index_collection = self.get_mean_index_collection(batch_image_collection, calculate_index)
            mean_index_fc = self.flatten_feature_collections(mean_index_collection)
            
            # Build dataframe from results
            df = self.get_results_as_dataframe([mean_index_fc], [index_name])
            
            self.logger.info(f"Batch {b} processed successfully: {len(df)} records for {index_name}")
            return df
        except Exception as e:
            self.logger.error(f"Error processing {index_name} for batch {b}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame in case of failure