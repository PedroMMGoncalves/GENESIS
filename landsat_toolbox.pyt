# -*- coding: utf-8 -*-

# landsat_toolbox.pyt
import arcpy
import os
from datetime import datetime
import arcpy.sa  
import arcpy.ia
import uuid
import numpy as np
from sklearn.decomposition import FastICA
import scipy.stats
from arcpy.ia import ExtractBand
from arcpy.sa import Float, Divide, Times, Con, SetNull, Plus, Minus
from arcpy.management import CompositeBands

class TransformStatistics:
    """Base class for transformation statistics"""
    def __init__(self):
        self.creation_date = datetime.now()
        self.description = ""
        self.errors = []
        
    def validate(self):
        """Base validation method"""
        return len(self.errors) == 0
        
    def save(self, filepath):
        """Save statistics to file"""
        try:
            directory = os.path.dirname(filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            np.savez(
                filepath,
                creation_date=self.creation_date,
                description=self.description,
                **self._get_save_dict()
            )
            return True
        except Exception as e:
            self.errors.append(f"Error saving statistics: {str(e)}")
            return False
            
    def load(self, filepath):
        """Load statistics from file"""
        try:
            if not os.path.exists(filepath):
                self.errors.append(f"Statistics file not found: {filepath}")
                return False
                
            data = np.load(filepath)
            self.creation_date = data['creation_date'].item()
            self.description = str(data['description'])
            return self._load_from_dict(data)
        except Exception as e:
            self.errors.append(f"Error loading statistics: {str(e)}")
            return False
            
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        raise NotImplementedError
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        raise NotImplementedError

class MNFNoiseStatistics(TransformStatistics):
    """Statistics from first MNF rotation (noise statistics)"""
    def __init__(self):
        super().__init__()
        self.noise_covariance = None
        self.noise_eigenvalues = None
        self.noise_eigenvectors = None
        self.description = "MNF Noise Statistics"
        
    def validate(self):
        """Validate noise statistics"""
        if not super().validate():
            return False
            
        if self.noise_covariance is None:
            self.errors.append("Noise covariance matrix is missing")
        if self.noise_eigenvalues is None:
            self.errors.append("Noise eigenvalues are missing")
        if self.noise_eigenvectors is None:
            self.errors.append("Noise eigenvectors are missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'noise_covariance': self.noise_covariance,
            'noise_eigenvalues': self.noise_eigenvalues,
            'noise_eigenvectors': self.noise_eigenvectors
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.noise_covariance = data['noise_covariance']
            self.noise_eigenvalues = data['noise_eigenvalues']
            self.noise_eigenvectors = data['noise_eigenvectors']
            return True
        except Exception as e:
            self.errors.append(f"Error loading noise statistics: {str(e)}")
            return False

class MNFStatistics(TransformStatistics):
    """Statistics from MNF transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transform_matrix = None
        self.noise_covariance = None
        self.whitening_matrix = None
        self.signal_covariance = None
        self.component_correlation = None
        self.description = "MNF Transform Statistics"

    def validate(self):
        """Validate MNF statistics"""
        if not super().validate():
            return False

        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.transform_matrix is None:
            self.errors.append("Transform matrix is missing")

        return len(self.errors) == 0

    def _get_save_dict(self):
        """Get dictionary of values to save"""
        save_dict = {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'transform_matrix': self.transform_matrix
        }

        # Add optional fields if they exist
        if hasattr(self, 'noise_covariance') and self.noise_covariance is not None:
            save_dict['noise_covariance'] = self.noise_covariance

        if hasattr(self, 'whitening_matrix') and self.whitening_matrix is not None:
            save_dict['whitening_matrix'] = self.whitening_matrix

        if hasattr(self, 'signal_covariance') and self.signal_covariance is not None:
            save_dict['signal_covariance'] = self.signal_covariance

        if hasattr(self, 'component_correlation') and self.component_correlation is not None:
            save_dict['component_correlation'] = self.component_correlation

        return save_dict

    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.transform_matrix = data['transform_matrix']

            # Load optional fields if available
            if 'noise_covariance' in data:
                self.noise_covariance = data['noise_covariance']

            if 'whitening_matrix' in data:
                self.whitening_matrix = data['whitening_matrix']

            if 'signal_covariance' in data:
                self.signal_covariance = data['signal_covariance']

            if 'component_correlation' in data:
                self.component_correlation = data['component_correlation']

            return True
        except Exception as e:
            self.errors.append(f"Error loading MNF statistics: {str(e)}")
            return False

class PCAStatistics(TransformStatistics):
    """Statistics for PCA transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance = None
        self.covariance_matrix = None
        self.description = "PCA Transform Statistics"
        
    def validate(self):
        """Validate PCA statistics"""
        if not super().validate():
            return False
            
        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.explained_variance is None:
            self.errors.append("Explained variance is missing")
        if self.covariance_matrix is None:
            self.errors.append("Covariance matrix is missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'explained_variance': self.explained_variance,
            'covariance_matrix': self.covariance_matrix
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.explained_variance = data['explained_variance']
            self.covariance_matrix = data['covariance_matrix']
            return True
        except Exception as e:
            self.errors.append(f"Error loading PCA statistics: {str(e)}")
            return False

class ICAStatistics(TransformStatistics):
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.mixing_matrix = None
        self.unmixing_matrix = None
        self.whitening_matrix = None
        self.dewhitening_matrix = None
        self.n_iterations = None
        self.independence_metrics = None
        self.kurtosis_values = None
        
    def validate(self):
        """Validate ICA statistics"""
        if not super().validate():
            return False
            
        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.mixing_matrix is None:
            self.errors.append("Mixing matrix is missing")
        if self.unmixing_matrix is None:
            self.errors.append("Unmixing matrix is missing")
        if self.whitening_matrix is None:
            self.errors.append("Whitening matrix is missing")
        if self.dewhitening_matrix is None:
            self.errors.append("Dewhitening matrix is missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'band_means': self.band_means,
            'mixing_matrix': self.mixing_matrix,
            'unmixing_matrix': self.unmixing_matrix,
            'whitening_matrix': self.whitening_matrix,
            'dewhitening_matrix': self.dewhitening_matrix,
            'n_iterations': self.n_iterations,
            'independence_metrics': self.independence_metrics
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.mixing_matrix = data['mixing_matrix']
            self.unmixing_matrix = data['unmixing_matrix']
            self.whitening_matrix = data['whitening_matrix']
            self.dewhitening_matrix = data['dewhitening_matrix']
            self.n_iterations = data['n_iterations']
            if 'independence_metrics' in data:
                self.independence_metrics = data['independence_metrics']
            return True
        except Exception as e:
            self.errors.append(f"Error loading ICA statistics: {str(e)}")
            return False


class Toolbox(object):
    def __init__(self):
        self.label = "Landsat Analysis Tools"
        self.alias = "landsattools"
        # Define all tools
        self.tools = [
            LandsatMosaicTool,          # Tool 1: Mosaic Creation
            LandsatIndicesComposite,    # Tool 2: Indices and Composites
            LandsatTransformations,     # Tool 3: Statistical Transformations
            LandsatSAM                  # Tool 4: Spectral Angle Mapper
        ]

# Tool 1: Mosaic Creation
class LandsatMosaicTool(object):
    def __init__(self):
        self.label = "Create Landsat Mosaic"
        self.description = "Creates mosaics from Landsat 8/9 scenes"
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Output Geodatabase
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        gdb.filter.list = ["Local Database"]

        # Output Mosaic Name
        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        # Landsat Data Folder
        data_folder = arcpy.Parameter(
            displayName="Landsat Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # Region
        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        region.filter.list = ["Portugal Mainland", 
                      "Azores Western (Flores, Corvo)", 
                      "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)", 
                      "Azores Eastern (São Miguel, Santa Maria)", 
                      "Madeira", 
                      "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)", 
                      "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
                      "Angola", 
                      "Mozambique"]

        # Time Filter Type
        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        time_type.filter.list = ["All Images", "Specific Year", "Month in Year", 
                                "Month All Years", "Season in Year", "Season All Years"]

        # Year (optional, enabled for year-specific options)
        year = arcpy.Parameter(
            displayName="Year",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Month (optional)
        month = arcpy.Parameter(
            displayName="Month",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        month.filter.list = list(range(1, 13))

        # Season (optional)
        season = arcpy.Parameter(
            displayName="Season",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Mask Feature
        mask = arcpy.Parameter(
            displayName="Mask Feature (Optional)",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # Save Statistics
        save_stats = arcpy.Parameter(
            displayName="Save Processing Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True

        params = [gdb, mosaic_name, data_folder, region, time_type, 
                 year, month, season, mask, save_stats]
        return params
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        if not parameters[2].altered:  # If data folder not set
            return

        # Scan data folder for available years when folder is set
        if parameters[2].value and parameters[2].altered:
            try:
                years = set()
                folder_path = parameters[2].valueAsText
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('_MTL.txt'):
                            # Parse date from filename (format: LC08_L2SP_204032_20240215_...)
                            parts = file.split('_')
                            if len(parts) >= 4:
                                try:
                                    date_str = parts[3]
                                    year = int(date_str[:4])
                                    years.add(year)
                                except ValueError:
                                    continue
                if years:
                    parameters[5].filter.list = sorted(list(years))
                    arcpy.AddMessage(f"Found years: {sorted(list(years))}")
            except Exception as e:
                arcpy.AddWarning(f"Error scanning years: {str(e)}")

        # Update season list based on region
        if parameters[3].value:  # If region is selected
            region = parameters[3].valueAsText
            if "Angola" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Mozambique" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Cape Verde" in region:
                parameters[7].filter.list = ["Dry", "Rainy", "Transition Dry-Wet", "Transition Wet-Dry"]    
            else:  # Temperate regions
                parameters[7].filter.list = ["Spring", "Summer", "Autumn", "Winter"]

        # Enable/disable time-based parameters
        if parameters[4].value:  # Time Filter Type
            time_type = parameters[4].valueAsText
            parameters[5].enabled = time_type in ["Specific Year", "Month in Year", "Season in Year"]
            parameters[6].enabled = time_type in ["Month in Year", "Month All Years"]
            parameters[7].enabled = time_type in ["Season in Year", "Season All Years"]

    def updateMessages(self, parameters):
            """Modify messages created by internal validation"""
            if parameters[0].altered:  # Geodatabase validation
                gdb_path = parameters[0].valueAsText
                arcpy.AddMessage(f"\nValidating geodatabase: {gdb_path}")
                
                try:
                    # Check if it's a geodatabase
                    if not gdb_path.endswith('.gdb'):
                        parameters[0].setErrorMessage("Output must be a File Geodatabase (.gdb)")
                        return

                    # Check if it exists
                    exists = arcpy.Exists(gdb_path)
                    arcpy.AddMessage(f"Geodatabase exists: {exists}")
                    if not exists:
                        parameters[0].setErrorMessage("Geodatabase does not exist")
                        return

                    # Verify it's a valid workspace
                    desc = arcpy.Describe(gdb_path)
                    arcpy.AddMessage(f"Workspace type: {desc.dataType}")
                    if desc.dataType != "Workspace":
                        parameters[0].setErrorMessage("Not a valid geodatabase workspace")
                        return

                    # Test write permissions
                    try:
                        test_name = "delete_me_test"
                        test_path = os.path.join(gdb_path, test_name)
                        arcpy.management.CreateFeatureclass(gdb_path, test_name, "POINT")
                        arcpy.management.Delete(test_path)
                        arcpy.AddMessage("Write permission test: Passed")
                    except Exception as e:
                        parameters[0].setErrorMessage(f"No write permissions: {str(e)}")
                        return

                except Exception as e:
                    parameters[0].setErrorMessage(f"Workspace validation error: {str(e)}")

            # Validate data folder
            if parameters[2].altered:
                folder_path = parameters[2].valueAsText
                if not os.path.exists(folder_path):
                    parameters[2].setErrorMessage("Data folder does not exist")
                    return

                # Check for Landsat scenes
                found_scene = False
                for root, _, files in os.walk(folder_path):
                    if any(f.endswith('_MTL.txt') for f in files):
                        found_scene = True
                        break

                if not found_scene:
                    parameters[2].setErrorMessage("No Landsat scenes found in folder")
                    
    def remove_cloud(self, scenes, stats):
        """Remove clouds from Landsat scenes"""
        try:
            from arcpy.ia import TransposeBits
            from arcpy.sa import Con  # More flexible than Clip
            
            start_time = datetime.now()
            self._update_processing_stats(stats, stage="cloud_removal")
            clean_scenes = []
            total_scenes = len(scenes)
            
            arcpy.AddMessage(f"\nRemoving clouds from {total_scenes} scenes...")
            
            for idx, scene in enumerate(scenes, 1):
                try:
                    scene_path = scene['path']
                    arcpy.AddMessage(f"\nProcessing scene {idx} of {total_scenes}")
                    arcpy.AddMessage(f"Scene path: {scene_path}")
                    
                    # List and check files
                    files = os.listdir(scene_path)
                    
                    # Find spectral bands and QA band
                    band_files = []
                    qa_file = None
                    
                    for file in files:
                        if '_SR_B' in file and file.endswith('.TIF'):
                            band_number = int(file.split('_SR_B')[-1].split('.')[0])
                            if 1 <= band_number <= 7:
                                band_files.append(os.path.join(scene_path, file))
                        
                        if '_QA_PIXEL.TIF' in file:
                            qa_file = os.path.join(scene_path, file)
                    
                    # Validate files
                    if not band_files or not qa_file:
                        arcpy.AddWarning(f"Incomplete scene data for {scene_path}")
                        continue
                    
                    # Sort band files to ensure correct order
                    band_files.sort()
                    
                    # Create rasters
                    band_rasters = [arcpy.Raster(f) for f in band_files]
                    qa_raster = arcpy.Raster(qa_file)
                    
                    # Create cloud mask
                    # Use the original TransposeBits signature
                    cloud_mask = TransposeBits(qa_raster, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 0, None)
                    
                    # Invert mask to keep clear pixels
                    value_mask = ~cloud_mask
                    
                    # Create clean rasters using Con (conditional) function
                    # This keeps pixels where value_mask is True (clear pixels)
                    clean_band_rasters = [Con(value_mask, raster) for raster in band_rasters]
                    
                    # Add processed scene to clean scenes
                    clean_scenes.append({
                        'path': scene_path,
                        'rasters': clean_band_rasters,
                        'metadata': scene.get('metadata', {})
                    })
                    
                    arcpy.AddMessage(f"Cloud removal completed for scene {idx}")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing scene {idx}: {str(e)}")
                    stats['failed_scenes'] = stats.get('failed_scenes', 0) + 1
                    stats['errors'].append(str(e))
                    continue
            
            # Update cloud removal statistics
            stats['cloud_removal'] = {
                'scenes_processed': total_scenes,
                'scenes_cleaned': len(clean_scenes),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            if not clean_scenes:
                arcpy.AddWarning("No scenes were successfully processed")
                return None
                
            return clean_scenes
            
        except Exception as e:
            arcpy.AddError(f"Cloud removal failed: {str(e)}")
            return None

    def _create_geometric_median_mosaic(self, clean_scenes, gdb_path, mosaic_name):
        """Create geometric median mosaic preserving multi-band structure"""
        try:
            start_time = datetime.now()
            
            # Create list of multi-band rasters
            multiband_rasters = []
            for scene in clean_scenes:
                if 'rasters' in scene and len(scene['rasters']) == 7:
                    # Create temporary composite
                    temp_composite = os.path.join(gdb_path, f"temp_composite_{uuid.uuid4().hex}")
                    arcpy.management.CompositeBands(scene['rasters'], temp_composite)
                    multiband_rasters.append(temp_composite)
            
            # Process all bands together
            output_path = os.path.join(gdb_path, f"{mosaic_name}_Geomedian")
            geomedian = arcpy.ia.GeometricMedian(
                multiband_rasters, 
                epsilon=0.001, 
                max_iteration=20, 
                extent_type="UnionOf", 
                cellsize_type="FirstOf"
            )
            geomedian.save(output_path)
            
            # Clean up temporary composites
            for temp_raster in multiband_rasters:
                if arcpy.Exists(temp_raster):
                    arcpy.management.Delete(temp_raster)
            
            # Update statistics
            stats = {
                'geometric_median': {
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'scenes_processed': len(multiband_rasters)
                }
            }
            
            arcpy.AddMessage(f"Multi-band geometric median created: {output_path}")
            return output_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating geometric median: {str(e)}")
            return None
                    
    def execute(self, parameters, messages):
        try:
            # Check out necessary extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            if arcpy.CheckExtension("ImageAnalyst") == "Available":
                arcpy.CheckOutExtension("ImageAnalyst")
            
            # Enable overwrite
            arcpy.env.overwriteOutput = True
            
            # Get parameters
            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            mask_feature = parameters[8].valueAsText
            save_stats = parameters[9].value

            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'total_scenes': 0,
                'processed_scenes': 0,
                'failed_scenes': 0,
                'cloud_coverage': [],
                'processing_time': [],
                'errors': []
            }

            # Prepare stats for cloud removal and geometric median
            stats['cloud_removal'] = {
                'scenes_processed': 0,
                'scenes_cleaned': 0,
                'processing_time': 0
            }
            stats['geometric_median'] = {
                'batches_processed': 0,
                'total_batches': 0,
                'processing_time': 0
            }

            arcpy.AddMessage("\nInitializing processing:")
            arcpy.AddMessage(f"Workspace: {gdb_path}")
            arcpy.AddMessage(f"Output name: {mosaic_name}")
            arcpy.AddMessage(f"Region: {region}")

            # Create temporal filter and get region info
            temporal_filter = self._create_temporal_filter(time_type, year, month, season)
            region_info = self._get_region_info(region)

            # Process each UTM zone
            final_mosaics = []
            for utm_zone in region_info['utm_zones']:
                try:
                    arcpy.AddMessage(f"\nProcessing UTM zone {utm_zone}{region_info['hemisphere']}")
                    
                    # Find scenes for this zone
                    scenes = self._find_scenes(
                        data_folder=data_folder,
                        utm_zone=utm_zone,
                        temporal_filter=temporal_filter,
                        seasonal_pattern=region_info['seasonal_pattern'],
                        stats=stats
                    )
                    
                    if not scenes:
                        arcpy.AddWarning(f"No scenes found for UTM zone {utm_zone}")
                        continue

                    # Remove clouds from scenes
                    arcpy.AddMessage("\nRemoving clouds from scenes...")
                    clean_scenes = self.remove_cloud(scenes, stats)
                    
                    if not clean_scenes:
                        arcpy.AddWarning(f"No valid scenes after cloud removal for UTM zone {utm_zone}")
                        continue

                    # Create geometric median mosaic
                    zone_mosaic = self._create_geometric_median_mosaic(
                        clean_scenes,
                        gdb_path,
                        f"{mosaic_name}_UTM{utm_zone}{region_info['hemisphere']}"
                    )
                    
                    if zone_mosaic:
                        final_mosaics.append(zone_mosaic)

                except Exception as e:
                    arcpy.AddWarning(f"Error processing UTM zone {utm_zone}: {str(e)}")
                    stats['errors'].append(f"Zone {utm_zone} error: {str(e)}")
                    continue

            # Check if any mosaics were created
            if not final_mosaics:
                arcpy.AddError("No valid mosaics were created")
                return None

            # Merge zones if needed
            if len(final_mosaics) > 1:
                arcpy.AddMessage("\nMerging UTM zones...")
                final_mosaic = self._merge_zone_mosaics(
                    gdb_path, mosaic_name, final_mosaics, region_info
                )
            else:
                final_mosaic = final_mosaics[0]

            # Apply mask if specified
            if final_mosaic and mask_feature:
                arcpy.AddMessage("\nApplying mask...")
                final_mosaic = self._apply_mask(
                    final_mosaic, mask_feature, gdb_path, mosaic_name
                )

            # Save statistics
            if save_stats:
                # Populate additional statistics
                stats['processed_scenes'] = sum(1 for scene in clean_scenes) if 'clean_scenes' in locals() else 0
                stats['failed_scenes'] = stats.get('failed_scenes', 0)
                
                stats['end_time'] = datetime.now()
                stats['total_duration'] = stats['end_time'] - stats['start_time']
                
                # Save both regular and enhanced statistics
                self._save_statistics(gdb_path, mosaic_name, stats)
                self._save_enhanced_statistics(gdb_path, mosaic_name, stats)

            if final_mosaic:
                arcpy.AddMessage(f"\nProcessing completed successfully!")
                arcpy.AddMessage(f"Output mosaic: {final_mosaic}")
                return final_mosaic
            else:
                arcpy.AddError("Failed to create final mosaic")
                return None

        except Exception as e:
            arcpy.AddError(f"Critical error in processing: {str(e)}")
            raise
        
        finally:
            # Check in extensions
            try:
                for ext in ["Spatial", "ImageAnalyst"]:
                    if arcpy.CheckExtension(ext) == "Available":
                        arcpy.CheckInExtension(ext)
            except:
                pass
        
    def _create_temporal_filter(self, time_type, year, month, season):
        """Create temporal filter dictionary"""
        filter_dict = {'type': time_type.lower().replace(' ', '_')}
        
        if year:
            filter_dict['year'] = year
        if month:
            filter_dict['month'] = month
        if season:
            filter_dict['season'] = season
            
        arcpy.AddMessage(f"Temporal filter: {filter_dict}")
        return filter_dict
        
    def _get_region_info(self, region):
        """Get region UTM zones and other information"""
        region_info = {
            'Portugal Mainland': {
                'utm_zones': [29],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Western (Flores, Corvo)': {
                'utm_zones': [25],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Eastern (São Miguel, Santa Maria)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Madeira': {
                'utm_zones': [28],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Cape Verde Western (Santo Antão, São Vicente, São Nicolau)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)': {
                'utm_zones': [27],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Angola': {
                'utm_zones': [32, 33, 34],
                'hemisphere': 'S',
                'seasonal_pattern': 'angola'
            },
            'Mozambique': {
                'utm_zones': [36, 37],
                'hemisphere': 'S',
                'seasonal_pattern': 'mozambique'
            }
        }
        
        if region not in region_info:
            raise ValueError(f"Unknown region: {region}")
            
        return region_info[region]
        
    def _create_zone_mosaic(self, gdb_path, name, utm_zone, hemisphere):
        """Create mosaic dataset for specific UTM zone"""
        try:
            mosaic_path = os.path.join(gdb_path, name)
            epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
            
            arcpy.AddMessage(f"Creating mosaic dataset: {name}")
            arcpy.AddMessage(f"EPSG: {epsg}")
            
            if arcpy.Exists(mosaic_path):
                arcpy.AddMessage("Removing existing mosaic dataset")
                arcpy.Delete_management(mosaic_path)
                
            # Create base mosaic dataset
            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=name,
                coordinate_system=epsg,
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED",
                product_definition="NONE"
            )
            
            # Add time field
            arcpy.AddMessage("Adding time field...")
            arcpy.management.AddField(
                in_table=mosaic_path,
                field_name="acquisitionDate",
                field_type="DATE",
                field_alias="Acquisition Date"
            )
            
            # Configure mosaic properties with proper time settings
            arcpy.AddMessage("Configuring mosaic properties...")
            arcpy.management.SetMosaicDatasetProperties(
                in_mosaic_dataset=mosaic_path,
                rows_maximum_imagesize=15000,
                columns_maximum_imagesize=15000,
                allowed_compressions="NONE",
                default_compression_type="NONE",
                resampling_type="BILINEAR",
                clip_to_footprints="CLIP",
                footprints_may_contain_nodata="FOOTPRINTS_MAY_CONTAIN_NODATA",
                clip_to_boundary="CLIP",
                color_correction="NOT_APPLY",
                allowed_mensuration_capabilities="BASIC",
                default_mensuration_capabilities="BASIC",
                allowed_mosaic_methods="Center;NorthWest;Nadir;LockRaster;ByAttribute;Seamline;None",
                default_mosaic_method="ByAttribute",
                order_field="acquisitionDate",
                order_base="1/1/1900 12:00:00 AM",
                sorting_order="Ascending",
                mosaic_operator="FIRST",
                blend_width=10,
                view_point_x=0,
                view_point_y=0,
                max_num_per_mosaic=50,
                cell_size_tolerance=0.8,
                cell_size=30,
                metadata_level="BASIC",
                transmission_fields="acquisitionDate",
                use_time="ENABLED"
            )
            
            
            arcpy.AddMessage(f"Mosaic dataset created successfully: {name}")
            return mosaic_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating zone mosaic: {str(e)}")
            return None
        
          
    def _add_scenes_to_mosaic(self, mosaic_path, scenes, stats):
        """Add scenes to mosaic dataset"""
        try:
            arcpy.AddMessage("\nAdding scenes to mosaic...")
            
            if not scenes:
                arcpy.AddError("No scenes to add to mosaic")
                return False
            
            # Group scenes by satellite type
            scenes_by_type = {}
            for scene in scenes:
                scene_type = 'Landsat 8' if 'LC08' in scene['path'] else 'Landsat 9'
                if scene_type not in scenes_by_type:
                    scenes_by_type[scene_type] = []
                scenes_by_type[scene_type].append(scene)
            
            # Add scenes by satellite type
            for sat_type, type_scenes in scenes_by_type.items():
                try:
                    arcpy.AddMessage(f"\nProcessing {sat_type} scenes")
                    
                    # Add each scene folder
                    for scene in type_scenes:
                        arcpy.AddMessage(f"Adding scene: {scene['path']}")
                        
                        arcpy.management.AddRastersToMosaicDataset(
                            in_mosaic_dataset=mosaic_path,
                            raster_type=sat_type,
                            input_path=scene['path'],
                            aux_inputs="ProcessingTemplate Multiband"
                        )
                    
                    arcpy.AddMessage(f"Successfully added {len(type_scenes)} {sat_type} scenes")
                    
                    stats['processed_scenes'] += len(type_scenes)
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing {sat_type} scenes: {str(e)}")
                    stats['failed_scenes'] += len(type_scenes)
                    stats['errors'].append(str(e))
            
            # Check number of rasters in mosaic dataset
            raster_count = int(arcpy.management.GetCount(mosaic_path).getOutput(0))
            arcpy.AddMessage(f"Number of rasters in mosaic dataset: {raster_count}")
            
            if raster_count == 0:
                arcpy.AddError("No rasters were successfully added to the mosaic dataset")
                return False
            
            # Build seamlines
            arcpy.AddMessage("Building seamlines...")
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=mosaic_path,
                sort_order="ASCENDING",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="BOTH"
            )
            
            return True
            
        except Exception as e:
            arcpy.AddError(f"Error adding scenes to mosaic: {str(e)}")
            return False
            
    def _merge_zone_mosaics(self, gdb_path, mosaic_name, zone_mosaics, region_info):
        """Merge multiple UTM zone mosaics"""
        try:
            if len(zone_mosaics) == 1:
                return zone_mosaics[0]
                
            arcpy.AddMessage("\nMerging zone mosaics...")
            merged_name = f"{mosaic_name}_Merged"
            merged_path = os.path.join(gdb_path, merged_name)
            
            # Create merged mosaic in WGS 84
            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=merged_name,
                coordinate_system=4326,  # WGS 84
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED"
            )
            
            # Add each zone mosaic
            for zone_mosaic in zone_mosaics:
                arcpy.AddMessage(f"Adding {os.path.basename(zone_mosaic)}")
                arcpy.management.AddRastersToMosaicDataset(
                    in_mosaic_dataset=merged_path,
                    raster_type="Raster Dataset",
                    input_path=zone_mosaic
                )
                
            # Build seamlines for merged result
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=merged_path,
                cell_size=30,
                sort_order="Closest_To_Center",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="LINER"
            )
            
            return merged_path
            
        except Exception as e:
            arcpy.AddError(f"Error merging zone mosaics: {str(e)}")
            return None
        
    def _apply_mask(self, mosaic_path, mask_feature, gdb_path, mosaic_name):
        """Apply mask to mosaic dataset"""
        try:
            if not mask_feature:
                arcpy.AddMessage("No mask feature provided. Skipping masking.")
                return mosaic_path

            arcpy.AddMessage("\nApplying mask...")
            masked_name = f"{mosaic_name}_Masked"
            masked_path = os.path.join(gdb_path, masked_name)
            
            # Ensure mask feature exists
            if not arcpy.Exists(mask_feature):
                arcpy.AddWarning(f"Mask feature {mask_feature} does not exist.")
                return mosaic_path
            
            # Extract by mask using Spatial Analyst
            from arcpy.sa import ExtractByMask
            
            # Extract by mask
            extracted = ExtractByMask(mosaic_path, mask_feature)
            
            # Save the extracted raster
            extracted.save(masked_path)
            
            arcpy.AddMessage(f"Masked mosaic saved as: {masked_path}")
            return masked_path
            
        except Exception as e:
            arcpy.AddError(f"Error applying mask: {str(e)}")
            return mosaic_path  # Return original mosaic if mask fails
            
    def _save_statistics(self, gdb_path, mosaic_name, stats):
        """Save processing statistics to file"""
        try:
            # Create statistics folder if it doesn't exist
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Mosaic Processing Statistics\n")
                f.write("===================================\n\n")
                
                f.write(f"Processing Start: {stats['start_time']}\n")
                f.write(f"Processing End: {stats['end_time']}\n")
                f.write(f"Total Duration: {stats['total_duration']}\n\n")
                
                f.write("Scene Statistics:\n")
                f.write(f"Total Scenes Found: {stats['total_scenes']}\n")
                f.write(f"Successfully Processed: {stats['processed_scenes']}\n")
                f.write(f"Failed Scenes: {stats['failed_scenes']}\n\n")
                
                if stats['cloud_coverage']:
                    avg_cloud = sum(stats['cloud_coverage']) / len(stats['cloud_coverage'])
                    f.write(f"Average Cloud Coverage: {avg_cloud:.2f}%\n\n")
                    
                if stats['processing_time']:
                    avg_time = sum(stats['processing_time']) / len(stats['processing_time'])
                    f.write(f"Average Processing Time per Scene: {avg_time:.2f} seconds\n\n")
                    
                if stats['errors']:
                    f.write("Errors Encountered:\n")
                    for error in stats['errors']:
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nStatistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving statistics: {str(e)}")
            return None
        
    def _update_processing_stats(self, stats, stage="general"):
        """Enhanced statistics tracking"""
        try:
            # Add new statistics categories if not present
            if 'cloud_removal' not in stats:
                stats['cloud_removal'] = {
                    'scenes_processed': 0,
                    'scenes_failed': 0,
                    'average_cloud_coverage_before': 0,
                    'average_cloud_coverage_after': 0,
                    'processing_time': 0
                }
                
            if 'geometric_median' not in stats:
                stats['geometric_median'] = {
                    'batches_processed': 0,
                    'total_batches': 0,
                    'memory_usage': [],
                    'processing_time': 0
                }
                
            if 'memory_tracking' not in stats:
                stats['memory_tracking'] = {
                    'peak_memory': 0,
                    'average_memory': 0,
                    'timestamps': []
                }
                
            # Update memory tracking
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            stats['memory_tracking']['timestamps'].append({
                'time': datetime.now(),
                'memory': current_memory,
                'stage': stage
            })
            stats['memory_tracking']['peak_memory'] = max(
                stats['memory_tracking']['peak_memory'],
                current_memory
            )
            
            return stats
            
        except Exception as e:
            arcpy.AddWarning(f"Error updating statistics: {str(e)}")
            return stats

    def _save_enhanced_statistics(self, gdb_path, mosaic_name, stats):
        """Save enhanced processing statistics"""
        try:
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_detailed_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Processing Detailed Statistics\n")
                f.write("====================================\n\n")
                
                # Basic processing info
                f.write("Processing Duration:\n")
                f.write(f"Start: {stats.get('start_time', 'N/A')}\n")
                f.write(f"End: {stats.get('end_time', datetime.now())}\n")
                f.write(f"Total Duration: {stats.get('total_duration', 'N/A')}\n\n")
                
                # Cloud removal statistics
                cloud_removal = stats.get('cloud_removal', {})
                f.write("Cloud Removal Statistics:\n")
                f.write(f"Scenes Processed: {cloud_removal.get('scenes_processed', 0)}\n")
                f.write(f"Scenes Cleaned: {cloud_removal.get('scenes_cleaned', 0)}\n")
                f.write(f"Failed Scenes: {stats.get('failed_scenes', 0)}\n")
                f.write(f"Processing Time: {cloud_removal.get('processing_time', 0):.2f} seconds\n\n")
                
                # Geometric median statistics
                geo_median = stats.get('geometric_median', {})
                f.write("Geometric Median Statistics:\n")
                f.write(f"Batches Processed: {geo_median.get('batches_processed', 0)}\n")
                f.write(f"Total Batches: {geo_median.get('total_batches', 0)}\n")
                f.write(f"Processing Time: {geo_median.get('processing_time', 0):.2f} seconds\n\n")
                
                # Errors
                if stats.get('errors'):
                    f.write("Errors Encountered:\n")
                    for error in stats.get('errors', []):
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nDetailed statistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving enhanced statistics: {str(e)}")
            return None
            
    def _parse_metadata(self, mtl_path):
        """Parse Landsat MTL file"""
        try:
            with open(mtl_path) as f:
                content = f.read()
                
            # Extract key metadata
            scene_info = {}
            
            # Get acquisition date
            date_line = [line for line in content.split('\n') 
                        if 'DATE_ACQUIRED' in line][0]
            scene_info['acquisition_date'] = datetime.strptime(
                date_line.split('=')[1].strip(),
                '%Y-%m-%d'
            )
            
            # Get cloud cover
            cloud_line = [line for line in content.split('\n') 
                         if 'CLOUD_COVER' in line][0]
            scene_info['cloud_cover'] = float(cloud_line.split('=')[1].strip())
            
            # Get UTM zone
            utm_line = [line for line in content.split('\n') 
                       if 'UTM_ZONE' in line][0]
            scene_info['utm_zone'] = int(utm_line.split('=')[1].strip())
            
            # Get processing level
            level_line = [line for line in content.split('\n') 
                         if 'PROCESSING_LEVEL' in line][0]
            scene_info['processing_level'] = level_line.split('=')[1].strip()
            
            return scene_info
            
        except Exception as e:
            arcpy.AddWarning(f"Error parsing metadata {mtl_path}: {str(e)}")
            return None
        
    def _find_scenes(self, data_folder, utm_zone, temporal_filter, seasonal_pattern, stats):
        """Find and validate Landsat scenes for specified UTM zone"""
        try:
            scenes = []
            ls8_count = 0
            ls9_count = 0
            arcpy.AddMessage("\nScanning for Landsat scenes...")
            
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith('_MTL.txt'):
                        stats['total_scenes'] += 1
                        
                        try:
                            # Parse metadata
                            mtl_path = os.path.join(root, file)
                            scene_info = self._parse_metadata(mtl_path)
                            
                            if scene_info:
                                # Count Landsat 8 and 9 scenes
                                if 'LC08' in file:
                                    ls8_count += 1
                                elif 'LC09' in file:
                                    ls9_count += 1
                                
                                # Check UTM zone
                                if scene_info['utm_zone'] == utm_zone:
                                    # Apply temporal filter
                                    if self._apply_temporal_filter(scene_info, temporal_filter, seasonal_pattern):
                                        scenes.append({
                                            'path': root,
                                            'metadata': scene_info
                                        })
                                        stats['cloud_coverage'].append(scene_info['cloud_cover'])
                                        
                        except Exception as e:
                            stats['failed_scenes'] += 1
                            stats['errors'].append(str(e))
                            continue
            
            # Print Landsat 8 and 9 scene counts
            arcpy.AddMessage(f"Total Landsat 8 scenes: {ls8_count}")
            arcpy.AddMessage(f"Total Landsat 9 scenes: {ls9_count}")
            arcpy.AddMessage(f"Found {len(scenes)} valid scenes for UTM zone {utm_zone}")
            
            return scenes
            
        except Exception as e:
            arcpy.AddError(f"Error finding scenes: {str(e)}")
            return []
                
    def _apply_temporal_filter(self, scene_info, temporal_filter, seasonal_pattern):
        """Apply temporal filter to scene"""
        try:
            filter_type = temporal_filter['type']
            date = scene_info['acquisition_date']
            
            if filter_type == 'all_images':
                return True
                
            elif filter_type == 'specific_year':
                return date.year == temporal_filter['year']
                
            elif filter_type == 'month_in_year':
                return (date.year == temporal_filter['year'] and 
                        date.month == temporal_filter['month'])
                
            elif filter_type == 'month_all_years':
                return date.month == temporal_filter['month']
                
            elif filter_type == 'season_in_year':
                if 'year' in temporal_filter:
                    if date.year != temporal_filter['year']:
                        return False
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            elif filter_type == 'season_all_years':
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            return False
            
        except Exception as e:
            arcpy.AddWarning(f"Error applying temporal filter: {str(e)}")
            return False
            
    def _is_in_season(self, date, pattern, season):
        """Check if date falls within specified season"""
        season_months = {
            'temperate': {
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'autumn': [9, 10, 11],
                'winter': [12, 1, 2]
            },
            'angola': {
                'rainy': [11, 12, 1, 2, 3, 4],
                'dry': [5, 6, 7, 8, 9, 10],
                'rainy_peak': [1, 2, 3],
                'dry_peak': [6, 7, 8]
            },
            'cape_verde': {
                'dry': [12, 1, 2, 3, 4, 5, 6],
                'rainy': [8, 9, 10],
                'transition_dry_wet': [7],
                'transition_wet_dry': [11]
            },
            'mozambique': {
                'rainy': [10, 11, 12, 1, 2, 3],
                'dry': [4, 5, 6, 7, 8, 9],
                'rainy_peak': [12, 1, 2],
                'dry_peak': [7, 8, 9]
            }
        }
        
        return date.month in season_months[pattern][season.lower()]   

# Tool 2: Indices and Composites
class LandsatIndicesComposite(object):
    def __init__(self):
        """Define the tool"""
        self.label = "Calculate Indices and Composites"
        self.description = "Calculates spectral indices and creates band composites"
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""
        # Input Raster
        input_raster = arcpy.Parameter(
            displayName="Input Raster",
            name="input_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )

        # Output Workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )

        # Processing type
        process_type = arcpy.Parameter(
            displayName="Processing Type",
            name="process_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        process_type.filter.list = ["Spectral Indices", "Color Composites"]
        process_type.value = "Spectral Indices"  # Default value

        # Indices selection - Changed to Optional
        indices = arcpy.Parameter(
            displayName="Select Indices",
            name="indices",
            datatype="GPString",
            parameterType="Optional",  # Changed from Required to Optional
            direction="Input",
            multiValue=True,
            enabled=True
        )
        indices.filter.list = [
            "Clay Minerals Index (CMI)",
            "Ferrous Minerals Index (FMI)",
            "Iron Oxide Index (IOI)",
            "Ferric Iron Ratio (FIR)",
            "Ferrous Iron Ratio (FER)",
            "Alteration Index (AI)",
            "Clay Ratio (CR)",
            "Silica Index (SI)",
            "Clay Minerals Ratio (7/5)",
            "Alunite-Kaolinite-Pyrophyllite (6/7)",
            "Iron Oxides (4/2)",
            "Ferrous Minerals (6/5)",
            "Advanced Argillic Alteration Index (AAI)",
            "Muscovite Index (MI)",
            "Chlorite Index (CI)",
            "Gossan Index (GI)",
            "NDVI",
            "NDWI",
            "NDBI",
            "NDMI"
        ]
        # Set default selection
        indices.value = indices.filter.list[0]

        # Composites selection - Changed to Optional
        composites = arcpy.Parameter(
            displayName="Select Composites",
            name="composites",
            datatype="GPString",
            parameterType="Optional",  # Changed from Required to Optional
            direction="Input",
            multiValue=True,
            enabled=False
        )
        composites.filter.list = [
            "Natural Color (4,3,2)",
            "False Color (5,4,3)",
            "SWIR Geology Composite (7,6,4)",
            "Clay Minerals Composite (7,5,6)",
            "Ferrous Minerals Composite (6,5,4)",
            "Lithological Composite (7,4,2)",
            "Hydrothermal Alteration Composite (7/6,4,2)",
            "Iron Oxide Composite (4/3,6/7,6/5)",
            "Enhanced Outcrop (7,5,2)",
            "Vegetation-Outcrop (5,6,7)",
            "Sabin's Ratio (4/2,6/7,6/5)",
            "Outcrop-Vegetation (4/2,6/7,5)",
            "Kaufmann Ratio (7/5,5/4,6/7)",
            "Band Ratio Composite (4/2,6/7,6/5)"
        ]
        # Set default selection
        composites.value = composites.filter.list[0]

        # Output prefix
        out_prefix = arcpy.Parameter(
            displayName="Output Prefix",
            name="out_prefix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )

        # Rescale option - DEFAULT CHANGED TO FALSE
        rescale = arcpy.Parameter(
            displayName="Rescale Outputs (0-255)",
            name="rescale",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        rescale.value = False  # Changed from True to False

        # Mask feature
        mask_feature = arcpy.Parameter(
            displayName="Mask Feature (Optional)",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        params = [input_raster, out_workspace, process_type, 
                indices, composites, out_prefix, rescale, mask_feature]
        return params

    def updateParameters(self, parameters):
        """Modify parameters before internal validation"""
        # Toggle between indices and composites
        if parameters[2].altered:
            if parameters[2].value == "Spectral Indices":
                parameters[3].enabled = True
                parameters[4].enabled = False
            else:  # Color Composites
                parameters[3].enabled = False
                parameters[4].enabled = True

        # Set default output prefix
        if parameters[0].altered and parameters[0].value and not parameters[5].altered:
            try:
                input_name = os.path.basename(parameters[0].valueAsText)
                name_parts = os.path.splitext(input_name)
                if name_parts[0]:
                    parameters[5].value = f"{name_parts[0]}_"
            except:
                pass

    def updateMessages(self, parameters):
        """Validate parameters based on enabled state"""
        # We add custom validation that checks if selections are made
        # only when the parameter is enabled
        
        if parameters[2].value == "Spectral Indices" and parameters[3].enabled:
            if not parameters[3].value:
                parameters[3].setErrorMessage("Please select at least one index")
        
        elif parameters[2].value == "Color Composites" and parameters[4].enabled:
            if not parameters[4].value:
                parameters[4].setErrorMessage("Please select at least one composite")

    def execute(self, parameters, messages):
        """Tool execution"""
        try:
            # Check out Spatial Analyst
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return

            # Enable overwrite
            arcpy.env.overwriteOutput = True

            # Get parameters
            input_raster = parameters[0].valueAsText
            out_workspace = parameters[1].valueAsText
            process_type = parameters[2].valueAsText
            
            # Get indices or composites based on processing type
            if process_type == "Spectral Indices":
                items_text = parameters[3].valueAsText if parameters[3].value else ""
                # Split and clean up any quotes from the items
                selected_items = [item.strip("'\"") for item in items_text.split(";")] if items_text else []
            else:  # Color Composites
                items_text = parameters[4].valueAsText if parameters[4].value else ""
                # Split and clean up any quotes from the items
                selected_items = [item.strip("'\"") for item in items_text.split(";")] if items_text else []
            
            out_prefix = parameters[5].valueAsText if parameters[5].value else ""
            rescale = parameters[6].value
            mask_feature = parameters[7].valueAsText if parameters[7].value else None

            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            
            # Initialize bands dictionary
            bands = {}
            
            # Extract bands using ExtractBand
            arcpy.AddMessage("Extracting bands...")
            for i in range(1, 8):  # Landsat has 7 bands
                try:
                    bands[i] = Float(ExtractBand(input_raster, [i]))
                    arcpy.AddMessage(f"Successfully extracted Band {i}")
                except Exception as e:
                    arcpy.AddWarning(f"Error extracting Band {i}: {str(e)}")
            
            # Check if we have enough bands
            if len(bands) < 7:
                arcpy.AddWarning(f"Only {len(bands)} bands were successfully extracted. Some indices/composites may fail.")
            
            # Create mask object if specified
            mask_obj = None
            if mask_feature:
                arcpy.AddMessage(f"Using mask: {mask_feature}")
                if arcpy.Exists(mask_feature):
                    desc = arcpy.Describe(mask_feature)
                    mask_obj = mask_feature  # Using the feature directly

            # Process based on selection
            if process_type == "Spectral Indices":
                self._calculate_indices(bands, selected_items, out_workspace, out_prefix, rescale, mask_obj)
            else:  # Color Composites
                self._create_composites(bands, selected_items, out_workspace, out_prefix, mask_obj)
                
            arcpy.AddMessage("Processing completed successfully!")
            
        except Exception as e:
            arcpy.AddError(f"Error in execution: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            
        finally:
            # Check in extension
            arcpy.CheckInExtension("Spatial")
    
    def _calculate_indices(self, bands, indices, out_workspace, out_prefix, rescale, mask_obj):
        """Calculate selected spectral indices"""
        from arcpy.sa import ExtractByMask
        
        arcpy.AddMessage("\nCalculating spectral indices:")
        
        # Dictionary with index calculations and descriptive names
        index_map = {
            "Clay Minerals Index (CMI)": {
                "func": lambda: Divide(bands[6], bands[7]),
                "formula": "B6/B7",
                "name": "Clay_Minerals"
            },
            "Ferrous Minerals Index (FMI)": {
                "func": lambda: Divide(bands[6], bands[4]),
                "formula": "B6/B4",
                "name": "Ferrous_Minerals"
            },
            "Iron Oxide Index (IOI)": {
                "func": lambda: Divide(bands[4], bands[3]),
                "formula": "B4/B3",
                "name": "Iron_Oxide"
            },
            "Ferric Iron Ratio (FIR)": {
                "func": lambda: Divide(bands[4], bands[2]),
                "formula": "B4/B2",
                "name": "Ferric_Iron"
            },
            "Ferrous Iron Ratio (FER)": {
                "func": lambda: Divide(bands[5], bands[4]),
                "formula": "B5/B4",
                "name": "Ferrous_Iron"
            },
            "Alteration Index (AI)": {
                "func": lambda: Times(Divide(bands[6], bands[7]), Divide(bands[4], bands[2])),
                "formula": "(B6/B7)*(B4/B2)",
                "name": "Alteration"
            },
            "Clay Ratio (CR)": {
                "func": lambda: Divide(Times(bands[5], bands[7]), Times(bands[6], bands[6])),
                "formula": "(B5*B7)/(B6*B6)",
                "name": "Clay_Ratio"
            },
            "Silica Index (SI)": {
                "func": lambda: Divide(bands[6], bands[5]),
                "formula": "B6/B5",
                "name": "Silica"
            },
            "Clay Minerals Ratio (7/5)": {
                "func": lambda: Divide(bands[7], bands[5]),
                "formula": "B7/B5",
                "name": "Clay_Minerals_Ratio"
            },
            "Alunite-Kaolinite-Pyrophyllite (6/7)": {
                "func": lambda: Divide(bands[6], bands[7]),
                "formula": "B6/B7",
                "name": "Alunite_Kaolinite"
            },
            "Iron Oxides (4/2)": {
                "func": lambda: Divide(bands[4], bands[2]),
                "formula": "B4/B2",
                "name": "Iron_Oxides"
            },
            "Ferrous Minerals (6/5)": {
                "func": lambda: Divide(bands[6], bands[5]),
                "formula": "B6/B5",
                "name": "Ferrous_Minerals"
            },
            "Advanced Argillic Alteration Index (AAI)": {
                "func": lambda: Divide(Minus(bands[4], bands[6]), Plus(bands[4], bands[6])),
                "formula": "(B4-B6)/(B4+B6)",
                "name": "Adv_Argillic"
            },
            "Muscovite Index (MI)": {
                "func": lambda: Divide(bands[7], bands[6]),
                "formula": "B7/B6",
                "name": "Muscovite"
            },
            "Chlorite Index (CI)": {
                "func": lambda: Divide(Minus(bands[6], bands[7]), Plus(bands[6], bands[7])),
                "formula": "(B6-B7)/(B6+B7)",
                "name": "Chlorite"
            },
            "Gossan Index (GI)": {
                "func": lambda: Times(Divide(bands[4], bands[2]), Divide(bands[4], bands[3])),
                "formula": "(B4/B2)*(B4/B3)",
                "name": "Gossan"
            },
            "NDVI": {
                "func": lambda: Divide(Minus(bands[5], bands[4]), Plus(bands[5], bands[4])),
                "formula": "(B5-B4)/(B5+B4)",
                "name": "NDVI"
            },
            "NDWI": {
                "func": lambda: Divide(Minus(bands[3], bands[5]), Plus(bands[3], bands[5])),
                "formula": "(B3-B5)/(B3+B5)",
                "name": "NDWI"
            },
            "NDBI": {
                "func": lambda: Divide(Minus(bands[6], bands[5]), Plus(bands[6], bands[5])),
                "formula": "(B6-B5)/(B6+B5)",
                "name": "NDBI"
            },
            "NDMI": {
                "func": lambda: Divide(Minus(bands[5], bands[6]), Plus(bands[5], bands[6])),
                "formula": "(B5-B6)/(B5+B6)",
                "name": "NDMI"
            }
        }
        
        # Process each selected index
        for index_name in indices:
            try:
                # Clean the index name from any extra quotes
                clean_index = index_name.strip("'\"")
                arcpy.AddMessage(f"\nProcessing: {clean_index}")
                
                # Get the calculation function
                if clean_index not in index_map:
                    arcpy.AddError(f"Index '{clean_index}' not found in calculation functions")
                    continue
                    
                index_info = index_map[clean_index]
                calculation_func = index_info["func"]
                
                # Calculate index
                index_result = calculation_func()
                
                # Apply mask if provided
                if mask_obj is not None:
                    index_result = ExtractByMask(index_result, mask_obj)
                
                # Clean up extreme values
                index_result = SetNull(Float(index_result) > 10000, index_result)
                index_result = SetNull(Float(index_result) < -10000, index_result)
                
                # Rescale to 0-255 if requested
                if rescale:
                    try:
                        # Get min/max values, being careful with locale settings
                        min_val = arcpy.GetRasterProperties_management(index_result, "MINIMUM").getOutput(0)
                        max_val = arcpy.GetRasterProperties_management(index_result, "MAXIMUM").getOutput(0)
                        
                        # Handle potential locale issues with commas vs periods
                        try:
                            min_val = float(min_val)
                        except ValueError:
                            min_val = float(min_val.replace(',', '.'))
                        
                        try:
                            max_val = float(max_val)
                        except ValueError:
                            max_val = float(max_val.replace(',', '.'))
                        
                        if max_val > min_val:
                            arcpy.AddMessage(f"  Rescaling from [{min_val:.4f}, {max_val:.4f}] to [0, 255]")
                            rescaled_result = 255 * (Float(index_result) - min_val) / (max_val - min_val)
                            index_result = rescaled_result
                    except Exception as e:
                        arcpy.AddWarning(f"  Could not rescale: {str(e)}")
                
                # Create a valid output name without special characters
                # Clean formula for valid filename
                safe_formula = f"{index_info['formula']}".replace('/', '_').replace('*', '_').replace('+', '_').replace('-', '_').replace('(', '').replace(')', '')
                safe_formula = safe_formula[:15]  # Truncate if too long
                descriptive_name = f"{out_prefix}{index_info['name']}_{safe_formula}"
                
                # Ensure output name is GDB-compliant
                descriptive_name = ''.join(c for c in descriptive_name if c.isalnum() or c == '_')
                if descriptive_name[0].isdigit():
                    descriptive_name = 'idx_' + descriptive_name
                    
                output_path = os.path.join(out_workspace, descriptive_name)
                
                # Save the result with alternative approach if needed
                arcpy.AddMessage(f"  Saving to: {output_path}")
                try:
                    index_result.save(output_path)
                except RuntimeError as e:
                    if "FGDBR" in str(e):
                        # Try saving to a temporary TIFF and then importing to geodatabase
                        temp_tiff = os.path.join(arcpy.env.scratchFolder, f"temp_{uuid.uuid4().hex}.tif")
                        arcpy.AddMessage(f"  Trying alternative save method via: {temp_tiff}")
                        index_result.save(temp_tiff)
                        arcpy.management.CopyRaster(temp_tiff, output_path)
                        # Clean up temp file
                        if os.path.exists(temp_tiff):
                            try:
                                os.remove(temp_tiff)
                            except:
                                pass
                    else:
                        # Re-raise other errors
                        raise
                
            except Exception as e:
                arcpy.AddWarning(f"Error calculating {index_name}: {str(e)}")
                import traceback
                arcpy.AddWarning(traceback.format_exc())
                continue
    
    def _create_composites(self, bands, composites, out_workspace, out_prefix, mask_obj):
        """
        Create color composites using a simplified, direct approach
        that avoids most temporary file operations
        """
        import os
        import numpy as np
        
        arcpy.AddMessage("\nCreating color composites:")
        
        # Define composite band assignments - simplified to fewer options that are known to work
        composite_map = {
            "Natural Color (4,3,2)": {
                "bands": [4, 3, 2],
                "name": "Natural_Color"
            },
            "False Color (5,4,3)": {
                "bands": [5, 4, 3],
                "name": "False_Color"
            },
            "SWIR Geology Composite (7,6,4)": {
                "bands": [7, 6, 4],
                "name": "SWIR_Geology"
            },
            "Clay Minerals Composite (7,5,6)": {
                "bands": [7, 5, 6],
                "name": "Clay_Minerals"
            },
            "Ferrous Minerals Composite (6,5,4)": {
                "bands": [6, 5, 4],
                "name": "Ferrous_Minerals"
            },
            "Lithological Composite (7,4,2)": {
                "bands": [7, 4, 2],
                "name": "Lithological"
            },
            "Enhanced Outcrop (7,5,2)": {
                "bands": [7, 5, 2],
                "name": "Enhanced_Outcrop"
            },
            "Vegetation-Outcrop (5,6,7)": {
                "bands": [5, 6, 7],
                "name": "Vegetation_Outcrop"
            }
        }
        
        # Ratio-based composites defined separately
        ratio_composites = {
            "Hydrothermal Alteration Composite (7/6,4,2)": {
                "bands": [(7, 6), 4, 2],  # Format: (numerator, denominator), simple band, simple band
                "name": "Hydrothermal_Alteration"
            },
            "Iron Oxide Composite (4/3,6/7,6/5)": {
                "bands": [(4, 3), (6, 7), (6, 5)],
                "name": "Iron_Oxide"
            },
            "Sabin's Ratio (4/2,6/7,6/5)": {
                "bands": [(4, 2), (6, 7), (6, 5)],
                "name": "Sabins_Ratio"
            },
            "Outcrop-Vegetation (4/2,6/7,5)": {
                "bands": [(4, 2), (6, 7), 5],
                "name": "Outcrop_Vegetation"
            },
            "Kaufmann Ratio (7/5,5/4,6/7)": {
                "bands": [(7, 5), (5, 4), (6, 7)],
                "name": "Kaufmann_Ratio"
            },
            "Band Ratio Composite (4/2,6/7,6/5)": {
                "bands": [(4, 2), (6, 7), (6, 5)],
                "name": "Band_Ratio"
            }
        }
        
        # Merge the two dictionaries for full processing
        for key, value in ratio_composites.items():
            composite_map[key] = value
        
        # Process each selected composite
        for composite_name in composites:
            try:
                # Clean the composite name from any extra quotes
                clean_composite = composite_name.strip("'\"")
                arcpy.AddMessage(f"\nProcessing: {clean_composite}")
                
                # Skip if not in our map
                if clean_composite not in composite_map:
                    arcpy.AddWarning(f"Composite '{clean_composite}' not found in processing definitions")
                    continue
                
                # Get composite info
                composite_info = composite_map[clean_composite]
                
                # Create output name
                descriptive_name = f"{out_prefix}{composite_info['name']}"
                descriptive_name = descriptive_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
                descriptive_name = ''.join(c for c in descriptive_name if c.isalnum() or c == '_')
                output_path = os.path.join(out_workspace, descriptive_name)
                
                # Check if this composite has simple bands or includes ratios
                has_ratios = any(isinstance(band, tuple) for band in composite_info["bands"])
                
                # Method 1: Direct composite approach for simple band composites
                if not has_ratios:
                    try:
                        # Get the required bands
                        rgb_bands = [bands[band_idx] for band_idx in composite_info["bands"]]
                        
                        # Apply mask if needed
                        if mask_obj is not None:
                            from arcpy.sa import ExtractByMask
                            rgb_bands = [ExtractByMask(band, mask_obj) for band in rgb_bands]
                        
                        # Direct composite bands function
                        arcpy.AddMessage(f"  Creating composite directly: {output_path}")
                        arcpy.management.CompositeBands(rgb_bands, output_path)
                        arcpy.AddMessage(f"  Successfully created {descriptive_name}")
                        continue
                        
                    except Exception as e:
                        arcpy.AddWarning(f"  Direct composite failed: {str(e)}")
                        # Fall through to alternative methods
                
                # Method 2: Use in-memory NumPy arrays for ratio-based composites or as fallback
                try:
                    arcpy.AddMessage(f"  Creating composite using NumPy arrays...")
                    
                    # Get sample properties from first band for reference
                    sample_band = next(iter(bands.values()))
                    sr = sample_band.spatialReference
                    extent = sample_band.extent
                    cell_size = (sample_band.meanCellWidth, sample_band.meanCellHeight)
                    width = sample_band.width
                    height = sample_band.height
                    
                    # Create arrays for each component
                    rgb_arrays = []
                    for band_spec in composite_info["bands"]:
                        if isinstance(band_spec, tuple):
                            # This is a band ratio
                            numerator_idx, denominator_idx = band_spec
                            numerator = bands[numerator_idx]
                            denominator = bands[denominator_idx]
                            
                            # Calculate ratio using map algebra
                            ratio = Divide(numerator, denominator)
                            
                            # Apply mask if needed
                            if mask_obj is not None:
                                from arcpy.sa import ExtractByMask
                                ratio = ExtractByMask(ratio, mask_obj)
                            
                            # Convert to array
                            array = arcpy.RasterToNumPyArray(ratio, nodata_to_value=0)
                            rgb_arrays.append(array)
                            
                        else:
                            # This is a simple band
                            band = bands[band_spec]
                            
                            # Apply mask if needed
                            if mask_obj is not None:
                                from arcpy.sa import ExtractByMask
                                band = ExtractByMask(band, mask_obj)
                            
                            # Convert to array
                            array = arcpy.RasterToNumPyArray(band, nodata_to_value=0)
                            rgb_arrays.append(array)
                    
                    # Check if all arrays were created successfully
                    if len(rgb_arrays) != 3:
                        arcpy.AddWarning(f"  Failed to create all component arrays")
                        continue
                    
                    # Create in-memory rasters directly from arrays
                    in_memory_rasters = []
                    for i, array in enumerate(rgb_arrays):
                        # Normalize array to prevent extremes (optional)
                        # Here we could scale the data to a common range if needed
                        
                        # Convert array to raster
                        raster = arcpy.NumPyArrayToRaster(
                            array,
                            arcpy.Point(extent.XMin, extent.YMin),
                            cell_size[0], cell_size[1]
                        )
                        
                        # Set the spatial reference
                        if sr:
                            arcpy.management.DefineProjection(raster, sr)
                        
                        # Add to list
                        in_memory_rasters.append(raster)
                    
                    # Create composite
                    arcpy.AddMessage(f"  Creating composite: {output_path}")
                    arcpy.management.CompositeBands(in_memory_rasters, output_path)
                    arcpy.AddMessage(f"  Successfully created {descriptive_name}")
                    
                except Exception as e:
                    arcpy.AddWarning(f"  NumPy array method failed: {str(e)}")
                    import traceback
                    arcpy.AddWarning(traceback.format_exc())
                    
                    # Last resort: simplify the bands and try a basic composite
                    try:
                        arcpy.AddMessage(f"  Trying simplified approach...")
                        
                        # Try with a simple 3-4-5 band composite as fallback
                        fallback_bands = [bands[i] for i in [5, 4, 3] if i in bands]
                        
                        if len(fallback_bands) == 3:
                            # Apply mask if needed
                            if mask_obj is not None:
                                from arcpy.sa import ExtractByMask
                                fallback_bands = [ExtractByMask(band, mask_obj) for band in fallback_bands]
                            
                            # Direct composite as last resort
                            arcpy.management.CompositeBands(fallback_bands, output_path)
                            arcpy.AddMessage(f"  Created {descriptive_name} with fallback bands (5,4,3)")
                        else:
                            arcpy.AddWarning(f"  Could not create composite {descriptive_name}")
                            
                    except Exception as fallback_error:
                        arcpy.AddWarning(f"  All methods failed for {descriptive_name}")
                        continue
                    
            except Exception as e:
                arcpy.AddWarning(f"Error creating {composite_name}: {str(e)}")
                import traceback
                arcpy.AddWarning(traceback.format_exc())
                continue

# Tool 3: Statistical Transformations   
class LandsatTransformations(object):
    """Tool for performing statistical transformations on Landsat imagery"""
    
    def __init__(self):
        self.label = "Statistical Transformations"
        self.description = "Performs MNF, PCA, and ICA transformations on Landsat 8/9 data"
        self.canRunInBackground = True
        
    def getParameterInfo(self):
        # Input raster bands
        input_rasters = arcpy.Parameter(
            displayName="Input Raster Bands",
            name="input_rasters",
            datatype=["DERasterDataset", "GPRasterLayer"],
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        
        # Transformation type
        transform_type = arcpy.Parameter(
            displayName="Transformation Type",
            name="transform_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        transform_type.filter.list = ["MNF", "PCA", "ICA"]
        
        # Number of components
        num_components = arcpy.Parameter(
            displayName="Number of Output Components",
            name="num_components",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        
        # Optional parameters for MNF
        noise_stats_file = arcpy.Parameter(
            displayName="Input Noise Statistics (Optional, MNF only)",
            name="noise_stats_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        noise_subset = arcpy.Parameter(
            displayName="Spatial Subset for Noise (Optional, MNF only)",
            name="noise_subset",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output name
        out_name = arcpy.Parameter(
            displayName="Output Name",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Save statistics
        save_stats = arcpy.Parameter(
            displayName="Save Transform Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True
        
        # Statistics folder
        stats_folder = arcpy.Parameter(
            displayName="Statistics Folder",
            name="stats_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Preserve Input Mask
        preserve_mask = arcpy.Parameter(
            displayName="Preserve Input Mask",
            name="preserve_mask",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        preserve_mask.value = True
        
        return [input_rasters, transform_type, num_components, 
                noise_stats_file, noise_subset,
                out_workspace, out_name, save_stats, stats_folder,
                preserve_mask]
        
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        try:
            # Get references to parameters for easier access
            input_rasters = parameters[0]
            transform_type = parameters[1]
            num_components = parameters[2]
            noise_stats_file = parameters[3]
            noise_subset = parameters[4]
            save_stats = parameters[7]
            stats_folder = parameters[8]
            
            # Enable/disable MNF-specific parameters based on transformation type
            if transform_type.altered:
                is_mnf = transform_type.valueAsText == "MNF"
                noise_stats_file.enabled = is_mnf
                noise_subset.enabled = is_mnf
                
                # If noise statistics file is provided, disable noise subset
                if is_mnf and noise_stats_file.altered and noise_stats_file.valueAsText:
                    noise_subset.enabled = False
            
            # Enable/disable statistics folder parameter based on save_stats
            if save_stats.altered:
                stats_folder.enabled = save_stats.value
            
            # If input rasters are set, update number of components
            if input_rasters.altered and input_rasters.value:
                try:
                    raster_paths = input_rasters.valueAsText.split(";")
                    total_bands = 0
                    
                    # Calculate total number of bands across all inputs
                    for raster_path in raster_paths:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    # Update number of components range
                    num_components.filter.type = "Range"
                    num_components.filter.list = [1, total_bands]
                    
                    # Set default number of components if not already set
                    if not num_components.altered:
                        num_components.value = min(3, total_bands)
                    
                    arcpy.AddMessage(f"Total bands available: {total_bands}")
                except Exception as e:
                    arcpy.AddWarning(f"Error counting bands: {str(e)}")
            
            # Set default stats folder if save_stats is enabled but folder not specified
            if save_stats.value and not stats_folder.altered:
                # Default to user documents
                default_folder = os.path.expanduser("~/Documents/ArcGIS/Statistics")
                if not os.path.exists(default_folder):
                    try:
                        os.makedirs(default_folder)
                    except:
                        pass
                
                if os.path.exists(default_folder):
                    stats_folder.value = default_folder
        
        except Exception as e:
            arcpy.AddWarning(f"Error updating parameters: {str(e)}")
            
    def updateMessages(self, parameters):
        """Validate parameters"""
        if parameters[0].altered:
            try:
                # Get input raster paths
                input_rasters = parameters[0].valueAsText.split(";")
                
                # For PCA, we'll be more flexible with inputs
                if parameters[1].valueAsText == "PCA":
                    # Just check that inputs exist and are rasters
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                else:
                    # For MNF or ICA, be more permissive but still check format
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                
                # Check that number of components doesn't exceed total bands
                if parameters[2].altered and parameters[0].altered:
                    total_bands = 0
                    for raster_path in input_rasters:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    if parameters[2].value > total_bands:
                        parameters[2].setErrorMessage(
                            f"Number of components ({parameters[2].value}) " +
                            f"cannot exceed total bands ({total_bands})"
                        )
                    
            except Exception as e:
                parameters[0].setErrorMessage(f"Error validating input: {str(e)}")
                
    def execute(self, parameters, messages):
        """Execute the tool"""
        try:
            # Check out Spatial Analyst extension
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return None
            
            # Enable overwrite output
            arcpy.env.overwriteOutput = True
            
            # Get parameters with updated indices
            input_rasters = parameters[0].valueAsText.split(";")
            transform_type = parameters[1].valueAsText
            num_components = parameters[2].value
            noise_stats_file = parameters[3].valueAsText
            noise_subset = parameters[4].valueAsText
            out_workspace = parameters[5].valueAsText
            out_name = parameters[6].valueAsText
            save_stats = parameters[7].value
            stats_folder_param = parameters[8].valueAsText if parameters[8].altered else None
            preserve_mask = parameters[9].value  # New preserve mask parameter
            
            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'transform_type': transform_type,
                'num_components': num_components,
                'errors': []
            }
            
            try:
                # Create or use specified statistics folder
                if save_stats:
                    if stats_folder_param:
                        stats_folder = stats_folder_param
                        # Ensure the folder exists
                        if not os.path.exists(stats_folder):
                            os.makedirs(stats_folder)
                    else:
                        # Default to user's documents folder if not specified
                        stats_folder = os.path.expanduser("~/Documents/ArcGIS/Statistics")
                        if not os.path.exists(stats_folder):
                            os.makedirs(stats_folder)
                    
                    # Create statistics file path with txt extension for all transforms
                    stats_file = os.path.join(stats_folder, f"{out_name}_{transform_type}_stats.txt")
                else:
                    stats_file = None
                    stats_folder = None
                
                # Output path for result
                out_path = os.path.join(out_workspace, out_name)
                
                # For PCA, use built-in Spatial Analyst function
                if transform_type == "PCA":
                    arcpy.AddMessage("Using Spatial Analyst PrincipalComponents tool...")
                    
                    try:
                        # Make sure workspace is set properly
                        original_workspace = arcpy.env.workspace
                        arcpy.env.workspace = out_workspace
                        
                        arcpy.AddMessage(f"Statistics will be saved to: {stats_file if save_stats else 'None'}")
                        
                        # Perform PCA analysis with simplified call
                        result = arcpy.sa.PrincipalComponents(
                            input_rasters,  # Just pass the list directly
                            num_components,
                            stats_file if save_stats else None
                        )
                        
                        # Save result explicitly
                        result.save(out_path)
                        
                        # Restore workspace
                        arcpy.env.workspace = original_workspace
                        
                        arcpy.AddMessage(f"PCA completed successfully. Output saved to: {out_path}")
                        return out_path
                        
                    except Exception as e:
                        arcpy.AddError(f"PCA failed: {str(e)}")
                        # Try to print extended error info
                        import traceback
                        arcpy.AddError(traceback.format_exc())
                        return None
                
                # For MNF and ICA, we need to load raster data
                arcpy.AddMessage("Using custom implementation...")
                
                # Process first raster only (we'll add multi-raster support later)
                if len(input_rasters) > 1:
                    arcpy.AddWarning("Multiple input rasters provided. Using only the first raster.")
                    
                raster_path = input_rasters[0]
                arcpy.AddMessage(f"Processing input: {os.path.basename(raster_path)}")
                
                # Verify raster exists
                if not arcpy.Exists(raster_path):
                    arcpy.AddError(f"Input raster does not exist: {raster_path}")
                    return None
                
                # Get raster properties
                raster_obj = arcpy.Raster(raster_path)
                band_count = raster_obj.bandCount
                arcpy.AddMessage(f"  Has {band_count} bands")
                arcpy.AddMessage(f"  Format: {raster_obj.format}")
                arcpy.AddMessage(f"  Width: {raster_obj.width}, Height: {raster_obj.height}")
                arcpy.AddMessage(f"  Data type: {raster_obj.pixelType}")
                
                # Get reference information
                extent = raster_obj.extent
                cell_size = (raster_obj.meanCellWidth, raster_obj.meanCellHeight)
                spatial_ref = raster_obj.spatialReference
                
                # Check if raster has a mask
                has_mask = False
                mask = None
                
                # Try different methods to find mask
                if hasattr(raster_obj, 'mask'):
                    mask = raster_obj.mask
                    has_mask = True
                    arcpy.AddMessage("  Raster has an explicit mask")
                elif hasattr(raster_obj, 'noDataValue'):
                    arcpy.AddMessage(f"  Raster has NoData value: {raster_obj.noDataValue}")
                    # We'll handle NoData during array processing
                else:
                    arcpy.AddMessage("  No mask or NoData value detected")
                
                # Store mask info in raster_info
                raster_info = {
                    'extent': extent,
                    'cell_size': cell_size,
                    'spatial_ref': spatial_ref,
                    'mask': mask if preserve_mask and has_mask else None
                }
                        
                # Load entire raster at once
                arcpy.AddMessage("  Using whole-raster loading approach...")
                
                try:
                    # Load the entire raster at once
                    data_array = arcpy.RasterToNumPyArray(raster_obj)
                    arcpy.AddMessage(f"  Full array loaded, shape: {data_array.shape}")
                    
                    # For multiband raster, the dimensions should be (bands, height, width)
                    # or (height, width, bands) depending on how ArcGIS returns it
                    if len(data_array.shape) == 3:
                        # Check if bands are in the first dimension
                        if data_array.shape[0] == band_count:
                            arcpy.AddMessage("  Transposing array from (bands, height, width) to (height, width, bands)")
                            data_array = np.transpose(data_array, (1, 2, 0))
                        else:
                            arcpy.AddMessage("  Array already in (height, width, bands) format")
                    else:
                        # Single band - reshape to add band dimension
                        arcpy.AddMessage("  Reshaping single band array")
                        data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
                    
                    arcpy.AddMessage(f"  Final array shape: {data_array.shape}")
                    
                    # Handle NoData values
                    arcpy.AddMessage("Handling NoData values...")
                    data_array = data_array.astype(float)
                    
                    # Check for NoData using a safer approach
                    try:
                        if hasattr(raster_obj, 'noDataValue') and raster_obj.noDataValue is not None:
                            no_data = raster_obj.noDataValue
                            for i in range(data_array.shape[2]):
                                data_array[:, :, i][data_array[:, :, i] == no_data] = np.nan
                            arcpy.AddMessage(f"Applied NoData value: {no_data}")
                        else:
                            arcpy.AddMessage("No explicit NoData value found, continuing with all data")
                    except Exception as e:
                        arcpy.AddWarning(f"Error handling NoData values: {str(e)}")
                        arcpy.AddMessage("Continuing without NoData handling")
                    
                    arcpy.AddMessage("Handled NoData values")
                    
                    # Perform transformation
                    arcpy.AddMessage(f"\nPerforming {transform_type} transformation...")
                    
                    if transform_type == "MNF":
                        # Process noise subset if provided
                        noise_data = None
                        if noise_subset:
                            arcpy.AddMessage(f"Extracting noise statistics from subset: {noise_subset}")
                            try:
                                # Create temporary raster from first band for extraction
                                temp_raster_path = os.path.join(arcpy.env.scratchWorkspace or "memory", "temp_raster")
                                temp_raster = arcpy.NumPyArrayToRaster(
                                    data_array[:,:,0],
                                    lower_left_corner=arcpy.Point(extent.XMin, extent.YMin),
                                    x_cell_size=cell_size[0],
                                    y_cell_size=cell_size[1]
                                )
                                temp_raster.save(temp_raster_path)
                                
                                # Extract by mask
                                subset = arcpy.sa.ExtractByMask(temp_raster_path, noise_subset)
                                
                                # Convert to numpy array
                                subset_array = arcpy.RasterToNumPyArray(subset)
                                
                                # If subset extraction was successful, prepare noise data
                                if subset_array.size > 0:
                                    noise_data = data_array.copy()
                                    arcpy.AddMessage(f"Using noise subset")
                                else:
                                    arcpy.AddWarning("Subset extraction yielded no valid pixels, using full image for noise estimation")
                                    
                                # Clean up
                                if arcpy.Exists(temp_raster_path) and "memory" not in temp_raster_path:
                                    arcpy.management.Delete(temp_raster_path)
                                    
                            except Exception as e:
                                arcpy.AddWarning(f"Error processing noise subset: {str(e)}")
                                arcpy.AddWarning("Using full image for noise estimation")
                                stats['errors'].append(f"Noise subset error: {str(e)}")
                        
                        transformed_data, transform_stats = self._perform_mnf(
                            data_array, 
                            num_components,
                            noise_stats_file,  # Existing noise stats file if provided 
                            noise_data,  # Subset data for noise estimation
                            stats
                        )
                    else:  # ICA
                        transformed_data, transform_stats = self._perform_ica(
                            data_array,
                            num_components,
                            stats
                        )
                    
                    # Create multiband output
                    output_path = self._create_multiband_output(
                        transformed_data,
                        raster_info,
                        out_workspace,
                        out_name,
                        preserve_mask
                    )
                    
                    # Define data_info for statistics files
                    data_info = {
                        'input_rasters': input_rasters,
                        'output_path': output_path
                    }
                    
                    # Save statistics in text format if requested
                    if save_stats:
                        arcpy.AddMessage(f"Saving statistics to: {stats_file}")
                        
                        if transform_type == "MNF":
                            self._save_mnf_statistics_txt(stats_file, data_info, transform_stats)
                        else:  # ICA
                            self._save_ica_statistics_txt(stats_file, data_info, transform_stats)
                    
                    # Update final statistics
                    stats['end_time'] = datetime.now()
                    stats['processing_time'] = (
                        stats['end_time'] - stats['start_time']
                    ).total_seconds()
                    
                    arcpy.AddMessage("\nProcessing completed successfully!")
                    arcpy.AddMessage(f"Total processing time: {stats['processing_time']:.2f} seconds")
                    
                    return output_path
                    
                except Exception as e:
                    arcpy.AddError(f"Error loading or processing raster: {str(e)}")
                    import traceback
                    arcpy.AddError(traceback.format_exc())
                    return None
                    
            except Exception as e:
                arcpy.AddError(f"Error in transformation: {str(e)}")
                stats['errors'].append(str(e))
                # Print detailed traceback for debugging
                import traceback
                arcpy.AddError(traceback.format_exc())
                raise
                
        finally:
            # Check in extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")
                
    def _perform_mnf(self, data: np.ndarray, n_components: int, 
                noise_stats_file: str, noise_data: np.ndarray, 
                stats: dict) -> tuple[np.ndarray, MNFStatistics]:
        """
        Perform MNF transformation
        
        Parameters:
        -----------
        data : np.ndarray
            Input array of shape (height, width, bands)
        n_components : int
            Number of components to extract
        noise_stats_file : str
            Path to existing noise statistics file (optional)
        noise_data : np.ndarray
            Subset of data to use for noise estimation (optional)
        stats : dict
            Dictionary to store processing statistics
            
        Returns:
        --------
        tuple[np.ndarray, MNFStatistics]
            Transformed data and MNF statistics
        """
        try:
            # Initialize statistics objects
            noise_stats = MNFNoiseStatistics()
            mnf_stats = MNFStatistics()
            
            # Load noise statistics if provided
            if noise_stats_file:
                arcpy.AddMessage(f"Loading noise statistics from: {noise_stats_file}")
                if not noise_stats.load(noise_stats_file):
                    arcpy.AddWarning("Failed to load noise statistics file, estimating noise instead")
                    noise_stats_file = None
            
            # If noise statistics weren't loaded, estimate them
            if not noise_stats_file:
                arcpy.AddMessage("Estimating noise statistics...")
                
                # Use noise subset data if provided, otherwise use full data
                noise_source = noise_data if noise_data is not None else data
                
                # Handle shape for noise data
                if len(noise_source.shape) == 3:
                    # Handle 3D data (height, width, bands)
                    noise_flat = noise_source.reshape(-1, noise_source.shape[-1])
                else:
                    # Handle 2D data (samples, bands)
                    noise_flat = noise_source
                
                # Remove NaN values
                valid_mask = ~np.isnan(noise_flat).any(axis=1)
                valid_noise = noise_flat[valid_mask]
                
                if valid_noise.shape[0] < noise_flat.shape[1] * 10:
                    arcpy.AddWarning(f"Low sample count for noise estimation: {valid_noise.shape[0]} samples " +
                                f"for {noise_flat.shape[1]} bands. Results may be unstable.")
                
                # Use shift difference method for noise estimation
                if noise_data is None:
                    arcpy.AddMessage("Using shift difference method for noise estimation...")
                    
                    # Reshape data back to original dimensions for spatial operations
                    height, width, bands = data.shape
                    
                    # Calculate horizontal and vertical differences
                    h_diff = np.zeros_like(data)
                    v_diff = np.zeros_like(data)
                    
                    # Compute differences
                    h_diff[:, 1:, :] = data[:, 1:, :] - data[:, :-1, :]
                    v_diff[1:, :, :] = data[1:, :, :] - data[:-1, :, :]
                    
                    # Flatten differences
                    h_diff_flat = h_diff.reshape(-1, bands)
                    v_diff_flat = v_diff.reshape(-1, bands)
                    
                    # Remove NaN values
                    h_valid_mask = ~np.isnan(h_diff_flat).any(axis=1)
                    v_valid_mask = ~np.isnan(v_diff_flat).any(axis=1)
                    
                    h_valid = h_diff_flat[h_valid_mask]
                    v_valid = v_diff_flat[v_valid_mask]
                    
                    # Combine valid difference samples
                    valid_diff_samples = np.vstack([h_valid, v_valid])
                    
                    # Calculate noise covariance
                    noise_covariance = np.cov(valid_diff_samples, rowvar=False) / 2
                else:
                    # Use direct covariance from noise subset
                    arcpy.AddMessage("Calculating noise statistics from subset data...")
                    noise_covariance = np.cov(valid_noise, rowvar=False)
                
                # Ensure noise covariance is positive definite
                min_eig = np.min(np.linalg.eigvalsh(noise_covariance))
                if min_eig <= 0:
                    arcpy.AddWarning("Noise covariance has non-positive eigenvalues. Adding regularization.")
                    noise_covariance += np.eye(noise_covariance.shape[0]) * abs(min_eig) * 1.1
                
                # Calculate noise eigendecomposition
                noise_eigenvals, noise_eigenvecs = np.linalg.eigh(noise_covariance)
                
                # Store noise statistics
                noise_stats.noise_covariance = noise_covariance
                noise_stats.noise_eigenvalues = noise_eigenvals
                noise_stats.noise_eigenvectors = noise_eigenvecs
                
                # Store in MNF statistics for later use
                mnf_stats.noise_covariance = noise_covariance
            
            # Flatten input data for transformation
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])
            
            # Handle NaN values
            valid_mask = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[valid_mask]
            
            arcpy.AddMessage(f"Processing {valid_data.shape[0]} valid pixels for MNF...")
            
            # Center the data
            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean
            
            # Perform noise whitening
            arcpy.AddMessage("Applying noise whitening transformation...")
            whitening_matrix = (noise_stats.noise_eigenvectors @ 
                            np.diag(1.0/np.sqrt(np.maximum(noise_stats.noise_eigenvalues, 1e-10))) @ 
                            noise_stats.noise_eigenvectors.T)
            
            # Apply whitening
            whitened_data = centered_data @ whitening_matrix
            
            # Calculate covariance of whitened data
            whitened_cov = np.cov(whitened_data.T)
            
            # Calculate eigendecomposition of whitened covariance
            signal_eigenvals, signal_eigenvecs = np.linalg.eigh(whitened_cov)
            
            # Sort in descending order
            idx = signal_eigenvals.argsort()[::-1]
            signal_eigenvals = signal_eigenvals[idx]
            signal_eigenvecs = signal_eigenvecs[:, idx]
            
            # Select components
            signal_eigenvals = signal_eigenvals[:n_components]
            signal_eigenvecs = signal_eigenvecs[:, :n_components]
            
            # Transform the data
            transformed_valid = whitened_data @ signal_eigenvecs
            
            # Calculate correlation between components
            component_correlation = np.corrcoef(transformed_valid.T)
            
            # Report eigenvalues (signal-to-noise ratios)
            arcpy.AddMessage("\nMNF Component Statistics (Signal-to-Noise Ratios):")
            for i in range(n_components):
                arcpy.AddMessage(f"  Component {i+1}: {signal_eigenvals[i]:.4f}")
            
            # Store MNF statistics
            mnf_stats.band_means = data_mean
            mnf_stats.eigenvalues = signal_eigenvals
            mnf_stats.eigenvectors = signal_eigenvecs
            mnf_stats.transform_matrix = whitening_matrix @ signal_eigenvecs
            mnf_stats.whitening_matrix = whitening_matrix
            mnf_stats.signal_covariance = whitened_cov
            mnf_stats.component_correlation = component_correlation
            
            # Reconstruct full data array
            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[valid_mask] = transformed_valid
            
            # Reshape to original spatial dimensions
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)
            
            return transformed_data, mnf_stats
            
        except Exception as e:
            stats['errors'].append(f"MNF Error: {str(e)}")
            raise

    def _perform_pca(self, data: np.ndarray, n_components: int, stats: dict) -> tuple[np.ndarray, PCAStatistics]:
        """
        Perform PCA transformation
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array of shape (height, width, bands)
        n_components : int
            Number of components to extract
        stats : dict
            Dictionary to store processing statistics
            
        Returns:
        --------
        tuple[np.ndarray, PCAStatistics]
            Transformed data and PCA statistics
        """
        try:
            # Initialize statistics object
            pca_stats = PCAStatistics()
            
            # Flatten data and center
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])
            
            # Handle NaN values
            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]
            
            arcpy.AddMessage(f"Computing PCA on {valid_data.shape[0]} valid pixels...")
            
            # Center the data
            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean
            
            # Calculate covariance matrix
            covariance_matrix = np.cov(centered_data, rowvar=False)
            
            # Calculate eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
            
            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Truncate to requested number of components
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
            
            # Calculate explained variance
            total_var = np.sum(eigenvals)
            explained_variance = eigenvals / total_var
            
            # Transform valid data
            transformed_valid = centered_data @ eigenvecs
            
            # Reconstruct full data array
            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[is_not_nan] = transformed_valid
            
            # Store PCA statistics
            pca_stats.band_means = data_mean
            pca_stats.eigenvalues = eigenvals
            pca_stats.eigenvectors = eigenvecs
            pca_stats.explained_variance = explained_variance
            pca_stats.covariance_matrix = covariance_matrix
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)
            arcpy.AddMessage("\nPCA Component Statistics:")
            for i in range(n_components):
                arcpy.AddMessage(f"  Component {i+1}: {explained_variance[i]*100:.2f}% variance " +
                            f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
            
            # Reshape back to original dimensions
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)
            
            return transformed_data, pca_stats
            
        except Exception as e:
            stats['errors'].append(f"PCA Error: {str(e)}")
            raise

    def _perform_ica(self, data: np.ndarray, n_components: int, stats: dict) -> tuple[np.ndarray, ICAStatistics]:
        """
        Perform ICA transformation with kurtosis metrics
        
        Parameters:
        -----------
        data : np.ndarray
            Input data array of shape (height, width, bands)
        n_components : int
            Number of components to extract
        stats : dict
            Dictionary to store processing statistics
            
        Returns:
        --------
        tuple[np.ndarray, ICAStatistics]
            Transformed data and ICA statistics
        """
        try:
            from sklearn.decomposition import FastICA
            
            # Initialize statistics object
            ica_stats = ICAStatistics()
            
            # Flatten the data for processing
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])
            
            # Handle NaN values
            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]
            
            arcpy.AddMessage(f"Processing {valid_data.shape[0]} valid pixels for ICA...")
            
            # Center the data
            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean
            
            # Compute covariance for whitening
            cov_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate whitening and dewhitening matrices
            whitening = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
            dewhitening = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            
            # Apply whitening
            whitened = centered_data @ whitening.T
            
            arcpy.AddMessage("Performing FastICA...")
            
            # Run FastICA
            ica = FastICA(
                n_components=n_components, 
                whiten=False,  # We already whitened
                max_iter=1000, 
                tol=1e-4,
                random_state=42  # For reproducibility
            )
            transformed_valid = ica.fit_transform(whitened)
                        
            # Calculate kurtosis for each component
            # Kurtosis measures the "peakedness" of the distribution
            # Higher absolute kurtosis suggests more independence
            kurtosis_values = [
                scipy.stats.kurtosis(transformed_valid[:, i], fisher=True)  # Use scipy.stats.kurtosis
                for i in range(n_components)
            ]
            
            # Calculate negentropy as proxy for independence
            independence_metrics = []
            for i in range(transformed_valid.shape[1]):
                component = transformed_valid[:, i]
                # Simplified negentropy approximation using kurtosis
                negentropy = np.mean((component - np.mean(component))**4) / (np.var(component)**2) - 3
                independence_metrics.append(negentropy)
            
            # Store in statistics object
            ica_stats.kurtosis_values = kurtosis_values
            ica_stats.independence_metrics = independence_metrics
        
            # Get unmixing and mixing matrices
            W = ica.components_
            unmixing = W @ whitening
            mixing = np.linalg.pinv(unmixing)
            
            # Store statistics
            ica_stats.band_means = data_mean
            ica_stats.mixing_matrix = mixing
            ica_stats.unmixing_matrix = unmixing
            ica_stats.whitening_matrix = whitening
            ica_stats.dewhitening_matrix = dewhitening
            ica_stats.n_iterations = ica.n_iter_
            
            # Reconstruct full data array
            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[is_not_nan] = transformed_valid
            
            arcpy.AddMessage(f"ICA completed in {ica.n_iter_} iterations")
            
            # Reshape back to original dimensions
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)
            
            return transformed_data, ica_stats
            
        except ImportError:
            arcpy.AddError("scikit-learn is not available. Please install scikit-learn to use ICA")
            stats['errors'].append("scikit-learn is not available")
            raise
        except Exception as e:
            stats['errors'].append(f"ICA Error: {str(e)}")
            raise
        
    def _create_multiband_output(self, component_arrays, raster_info, out_workspace, out_name, preserve_mask=True):
        """
        Create a multiband raster from component arrays
        
        Parameters:
        -----------
        component_arrays : ndarray
            Component arrays of shape (height, width, components)
        raster_info : dict
            Raster information containing extent, cell size, etc.
        out_workspace : str
            Output workspace path
        out_name : str
            Output raster name
        preserve_mask : bool
            Whether to preserve the input mask
            
        Returns:
        --------
        str
            Path to the output multiband raster
        """
        try:
            arcpy.AddMessage("\nCreating multiband output...")
            
            # Extract raster information
            extent = raster_info['extent']
            cell_size = raster_info['cell_size']
            spatial_ref = raster_info['spatial_ref']
            mask = raster_info.get('mask', None)
            
            # Paths for temporary component rasters
            temp_component_paths = []
            
            # Create output directory for temp files if needed
            temp_dir = os.path.join(arcpy.env.scratchFolder, "temp_components")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Save each component as a temporary raster
            num_components = component_arrays.shape[2]
            for i in range(num_components):
                temp_path = os.path.join(temp_dir, f"temp_comp_{i+1}_{uuid.uuid4().hex}.tif")
                
                # Create raster from array
                out_raster = arcpy.NumPyArrayToRaster(
                    in_array=component_arrays[:,:,i],
                    lower_left_corner=arcpy.Point(
                        extent.XMin, 
                        extent.YMin
                    ),
                    x_cell_size=cell_size[0],
                    y_cell_size=cell_size[1],
                    value_to_nodata=np.nan
                )
                
                # Apply mask if requested and available
                if preserve_mask and mask is not None:
                    arcpy.AddMessage(f"  Applying mask to component {i+1}")
                    try:
                        # Convert mask to raster if it's not already
                        if not isinstance(mask, arcpy.Raster):
                            mask_raster = arcpy.Raster(mask)
                        else:
                            mask_raster = mask
                            
                        # Create SetNull expression
                        masked_raster = arcpy.sa.SetNull(mask_raster == 0, out_raster)
                        masked_raster.save(temp_path)
                    except Exception as e:
                        arcpy.AddWarning(f"  Error applying mask to component {i+1}: {str(e)}")
                        out_raster.save(temp_path)
                else:
                    # Save without masking
                    out_raster.save(temp_path)
                
                # Set spatial reference
                arcpy.DefineProjection_management(
                    temp_path,
                    spatial_ref
                )
                
                temp_component_paths.append(temp_path)
                arcpy.AddMessage(f"  Created component {i+1}")
            
            # Create multiband raster using Composite Bands
            output_path = os.path.join(out_workspace, out_name)
            arcpy.AddMessage(f"Creating final multiband raster: {output_path}")
            arcpy.management.CompositeBands(temp_component_paths, output_path)
            
            # Clean up temporary files
            arcpy.AddMessage("Cleaning up temporary files...")
            for temp_path in temp_component_paths:
                try:
                    arcpy.management.Delete(temp_path)
                except:
                    pass
            
            return output_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating multiband output: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _save_mnf_statistics_txt(self, stats_file, data_info, mnf_stats):
        """
        Save MNF statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        mnf_stats : MNFStatistics
            MNF statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Minimum Noise Fraction (MNF) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {len(mnf_stats.eigenvalues)}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n\n")
                
                # Noise Covariance Matrix
                f.write("#                    NOISE COVARIANCE MATRIX\n\n")
                num_bands = len(mnf_stats.band_means)
                f.write("#    Layer         " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                # Include noise covariance if available
                noise_cov = mnf_stats.noise_covariance if hasattr(mnf_stats, 'noise_covariance') else None
                if noise_cov is not None:
                    for i in range(noise_cov.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in noise_cov[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Noise Whitening Matrix
                if hasattr(mnf_stats, 'whitening_matrix') and mnf_stats.whitening_matrix is not None:
                    f.write("#                    NOISE WHITENING MATRIX\n\n")
                    f.write("#    Output Band   " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.whitening_matrix.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.whitening_matrix[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal Covariance Matrix (after whitening)
                if hasattr(mnf_stats, 'signal_covariance') and mnf_stats.signal_covariance is not None:
                    f.write("#                    SIGNAL COVARIANCE MATRIX (AFTER WHITENING)\n\n")
                    f.write("#    Band          " + "".join([f"{i+1:14d}" for i in range(mnf_stats.signal_covariance.shape[0])]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.signal_covariance.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.signal_covariance[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal-to-Noise Ratio (Eigenvalues)
                f.write("#                 SIGNAL-TO-NOISE RATIOS (EIGENVALUES)\n\n")
                f.write("# Number of Input Layers     Number of MNF Component Layers\n")
                f.write(f"            {len(mnf_stats.band_means):d}                              {len(mnf_stats.eigenvalues):d}\n")
                f.write("# MNF Layer        " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                f.write("# Eigenvalues\n")
                f.write("               " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvalues]) + "\n")
                
                # Eigenvectors
                f.write("# Eigenvectors\n")
                f.write("# Input Layer\n")
                for i in range(mnf_stats.eigenvectors.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvectors[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Correlation Matrix
                if hasattr(mnf_stats, 'component_correlation') and mnf_stats.component_correlation is not None:
                    f.write("#                 COMPONENT CORRELATION MATRIX\n")
                    f.write("# This matrix should be close to an identity matrix (diagonal = 1, off-diagonal ≈ 0)\n\n")
                    f.write("# Component      " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.component_correlation.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.component_correlation[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Percent and Accumulative Eigenvalues
                f.write("#                 PERCENT AND ACCUMULATIVE EIGENVALUES\n\n")
                f.write("# MNF Layer   EigenValue   Percent of EigenValues   Accumulative of EigenValues\n")
                
                total_eigenvalue = sum(mnf_stats.eigenvalues)
                cumulative = 0.0
                for i, eigenvalue in enumerate(mnf_stats.eigenvalues):
                    percent = (eigenvalue / total_eigenvalue) * 100.0
                    cumulative += percent
                    f.write(f"        {i+1:d} {eigenvalue:14.5f}          {percent:7.4f}               {cumulative:7.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving MNF statistics to text file: {str(e)}")

    def _save_ica_statistics_txt(self, stats_file, data_info, ica_stats):
        """
        Save ICA statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        ica_stats : ICAStatistics
            ICA statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Independent Component Analysis (ICA) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {ica_stats.mixing_matrix.shape[1]}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n")
                f.write(f"#\tConverged in {ica_stats.n_iterations} iterations\n\n")
                
                # Mixing Matrix
                f.write("#                    MIXING MATRIX (A)\n")
                f.write("# Independent components are reconstructed as X = AS, where A is the mixing matrix and S are the sources\n\n")
                f.write("#    Component    " + "".join([f"{i+1:14d}" for i in range(ica_stats.mixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.mixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.mixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Unmixing Matrix
                f.write("#                    UNMIXING MATRIX (W)\n")
                f.write("# Sources are computed as S = WX, where W is the unmixing matrix and X is the data\n\n")
                f.write("#    Source       " + "".join([f"{i+1:14d}" for i in range(ica_stats.unmixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.unmixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.unmixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Independence (approximate mutual information)
                f.write("#                 COMPONENT INDEPENDENCE MEASURES\n\n")
                f.write("# Component   Mutual Information   Independence Score\n")
                
                # Calculate simple independence metrics
                # In actual ICA implementation, mutual information would be calculated
                # Here we're providing placeholder values
                if hasattr(ica_stats, 'independence_metrics') and ica_stats.independence_metrics is not None:
                    for i, metric in enumerate(ica_stats.independence_metrics):
                        indep_score = 100 * (1 - np.exp(-metric))  # Convert to 0-100 scale
                        f.write(f"        {i+1:d}         {metric:8.4f}           {indep_score:8.4f}\n")
                else:
                    # Fallback if metrics aren't available
                    for i in range(ica_stats.mixing_matrix.shape[1]):
                        f.write(f"        {i+1:d}         {'N/A':8s}           {'N/A':8s}\n")
                f.write("#  ==========================================================================\n")
                              
                # Kurtosis Values
                f.write("# COMPONENT KURTOSIS VALUES\n")
                f.write("# Component   Kurtosis\n")
                for i, kurt in enumerate(ica_stats.kurtosis_values):
                    f.write(f"        {i+1:d}         {kurt:8.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving ICA statistics to text file: {str(e)}")

    def _extract_subset_data(self, data: np.ndarray, subset_feature: str) -> np.ndarray:
        """
        Extract data from specified spatial subset
        """
        try:
            # Create temporary raster from numpy array
            temp_raster = arcpy.NumPyArrayToRaster(data[:,:,0])
            
            # Extract by mask
            subset = arcpy.sa.ExtractByMask(temp_raster, subset_feature)
            
            # Convert back to numpy array
            subset_data = arcpy.RasterToNumPyArray(subset)
            
            return subset_data
            
        except Exception as e:
            arcpy.AddWarning(f"Error extracting subset: {str(e)}")
            return None

# Tool 4: Spectral Angle Mapper
class LandsatSAM(object):
    def __init__(self):
        self.label = "Spectral Angle Mapper"
        self.description = "Performs Spectral Angle Mapper classification on Landsat imagery"
        self.canRunInBackground = True
        
    def getParameterInfo(self):
        # Input raster
        input_raster = arcpy.Parameter(
            displayName="Input Landsat Raster",
            name="input_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Reference spectra source
        ref_source = arcpy.Parameter(
            displayName="Reference Spectra Source",
            name="ref_source",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        ref_source.filter.list = ["Table", "ROIs/Training Samples", "Endmember Pixels"]
        ref_source.value = "Table"
        
        # Reference spectra table (for Table option)
        ref_table = arcpy.Parameter(
            displayName="Reference Spectra Table",
            name="ref_table",
            datatype="DETable",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Training samples (for ROIs option)
        training_samples = arcpy.Parameter(
            displayName="Training Samples/ROIs",
            name="training_samples",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Endmember pixels (for Endmember option)
        endmember_pixels = arcpy.Parameter(
            displayName="Endmember Pixels (x,y coordinates)",
            name="endmember_pixels",
            datatype="GPValueTable",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        endmember_pixels.columns = [['GPString', 'Class Name'], ['GPLong', 'X'], ['GPLong', 'Y']]
        
        # Class names field (for Table option)
        class_field = arcpy.Parameter(
            displayName="Class Names Field",
            name="class_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        class_field.parameterDependencies = [ref_table.name]
        class_field.filter.list = ['Text']
        
        # Band fields (for Table option)
        band_fields = arcpy.Parameter(
            displayName="Band Value Fields",
            name="band_fields",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            enabled=True
        )
        band_fields.parameterDependencies = [ref_table.name]
        band_fields.filter.list = ['Double', 'Float', 'Long', 'Short']
        
        # Maximum angle
        max_angle = arcpy.Parameter(
            displayName="Maximum Angle (degrees)",
            name="max_angle",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        max_angle.value = 10.0
        max_angle.filter.type = "Range"
        max_angle.filter.list = [0.1, 90.0]
        
        # Threshold
        threshold = arcpy.Parameter(
            displayName="Classification Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        threshold.value = 0.1
        threshold.filter.type = "Range"
        threshold.filter.list = [0.01, 1.0]
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output classification raster
        out_raster = arcpy.Parameter(
            displayName="Output Classification Raster",
            name="out_raster",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Output SAM raster
        out_sam = arcpy.Parameter(
            displayName="Output SAM Angle Raster",
            name="out_sam",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        
        # Color scheme
        color_scheme = arcpy.Parameter(
            displayName="Apply Color Scheme",
            name="color_scheme",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        color_scheme.value = True
        
        return [input_raster, ref_source, ref_table, training_samples, endmember_pixels, 
                class_field, band_fields, max_angle, threshold, out_workspace, 
                out_raster, out_sam, color_scheme]
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        # Get references to parameters
        ref_source = parameters[1]
        ref_table = parameters[2]
        training_samples = parameters[3]
        endmember_pixels = parameters[4]
        class_field = parameters[5]
        band_fields = parameters[6]
        
        # Enable/disable parameters based on reference source
        if ref_source.altered:
            if ref_source.value == "Table":
                ref_table.enabled = True
                training_samples.enabled = False
                endmember_pixels.enabled = False
                class_field.enabled = True
                band_fields.enabled = True
            elif ref_source.value == "ROIs/Training Samples":
                ref_table.enabled = False
                training_samples.enabled = True
                endmember_pixels.enabled = False
                class_field.enabled = False
                band_fields.enabled = False
            elif ref_source.value == "Endmember Pixels":
                ref_table.enabled = False
                training_samples.enabled = False
                endmember_pixels.enabled = True
                class_field.enabled = False
                band_fields.enabled = False
        
        # Update output name based on input
        if parameters[0].altered and not parameters[10].altered:
            try:
                input_name = os.path.basename(parameters[0].valueAsText)
                name_parts = os.path.splitext(input_name)
                if name_parts[0]:
                    parameters[10].value = f"{name_parts[0]}_SAM_class"
                    parameters[11].value = f"{name_parts[0]}_SAM_angles"
            except:
                pass
        
    def updateMessages(self, parameters):
        """Modify messages created by internal validation"""
        # Validate reference spectra table
        if parameters[1].value == "Table" and parameters[2].altered:
            try:
                table_path = parameters[2].valueAsText
                if not arcpy.Exists(table_path):
                    parameters[2].setErrorMessage("Reference spectra table does not exist")
                else:
                    # Check for required fields
                    if parameters[5].value and parameters[6].value:
                        # Check class field
                        class_field = parameters[5].valueAsText
                        
                        # Check band fields (ensure they match input raster bands)
                        band_fields = parameters[6].valueAsText.split(";")
                        if len(band_fields) < 2:
                            parameters[6].setWarningMessage("At least 2 band fields should be specified")
            except Exception as e:
                parameters[2].setErrorMessage(f"Error validating reference table: {str(e)}")
        
        # Validate training samples
        if parameters[1].value == "ROIs/Training Samples" and parameters[3].altered:
            try:
                training_path = parameters[3].valueAsText
                if not arcpy.Exists(training_path):
                    parameters[3].setErrorMessage("Training samples do not exist")
                else:
                    # Check for class field
                    desc = arcpy.Describe(training_path)
                    if not any(field.name.lower() in ["class", "classname", "class_name"] for field in desc.fields):
                        parameters[3].setWarningMessage("No class field found. Expected fields: Class, ClassName, or Class_Name")
            except Exception as e:
                parameters[3].setErrorMessage(f"Error validating training samples: {str(e)}")
    
    def execute(self, parameters, messages):
        """Execute the tool"""
        try:
            # Check out extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return
                
            # Enable overwrite
            arcpy.env.overwriteOutput = True
            
            # Get parameters
            input_raster = parameters[0].valueAsText
            ref_source = parameters[1].valueAsText
            ref_table = parameters[2].valueAsText if parameters[1].value == "Table" else None
            training_samples = parameters[3].valueAsText if parameters[1].value == "ROIs/Training Samples" else None
            endmember_pixels = parameters[4].value if parameters[1].value == "Endmember Pixels" else None
            class_field = parameters[5].valueAsText if parameters[1].value == "Table" and parameters[5].value else None
            band_fields = parameters[6].valueAsText.split(";") if parameters[1].value == "Table" and parameters[6].value else []
            max_angle = parameters[7].value
            threshold = parameters[8].value
            out_workspace = parameters[9].valueAsText
            out_raster = parameters[10].valueAsText
            out_sam = parameters[11].valueAsText if parameters[11].value else None
            apply_color = parameters[12].value
            
            # Convert max angle to radians
            max_angle_rad = max_angle * (3.14159265359 / 180.0)
            
            # Process based on reference source
            if ref_source == "Table":
                # Perform SAM with reference table
                self._sam_with_table(
                    input_raster=input_raster,
                    ref_table=ref_table,
                    class_field=class_field,
                    band_fields=band_fields,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "ROIs/Training Samples":
                # Perform SAM with training samples
                self._sam_with_training(
                    input_raster=input_raster,
                    training_samples=training_samples,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "Endmember Pixels":
                # Perform SAM with endmember pixels
                self._sam_with_endmembers(
                    input_raster=input_raster,
                    endmember_pixels=endmember_pixels,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            
            # Return output path
            out_path = os.path.join(out_workspace, out_raster)
            arcpy.SetParameterAsText(10, out_path)
            
            if out_sam:
                sam_path = os.path.join(out_workspace, out_sam)
                arcpy.SetParameterAsText(11, sam_path)
                
            return out_path
            
        except Exception as e:
            arcpy.AddError(f"Error executing SAM: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
            
        finally:
            # Check in extensions
            arcpy.CheckInExtension("Spatial")
    
    def _sam_with_table(self, input_raster, ref_table, class_field, band_fields,
                        max_angle_rad, threshold, out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using reference spectra from a table"""
        import numpy as np
        import math
        
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Reference spectra table: {ref_table}")
            
            # Load the input raster
            raster_obj = arcpy.Raster(input_raster)
            band_count = raster_obj.bandCount
            
            # Check if band fields match available bands
            if len(band_fields) > band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Using only the available bands from the reference table")
                band_fields = band_fields[:band_count]
            elif len(band_fields) < band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Some raster bands will not be used in the analysis")
            
            # Load band data into memory
            arcpy.AddMessage("Loading raster bands...")
            bands = {}
            for i in range(1, band_count + 1):
                if i <= len(band_fields):  # Only load bands that have reference data
                    arcpy.AddMessage(f"Loading band {i}...")
                    bands[i] = arcpy.Raster(f"{input_raster}/{i}")
            
            # Read reference spectra from the table
            arcpy.AddMessage("Reading reference spectra...")
            ref_spectra = {}
            class_names = []
            
            with arcpy.da.SearchCursor(ref_table, [class_field] + band_fields) as cursor:
                for row in cursor:
                    class_name = row[0]
                    band_values = [float(row[i+1]) for i in range(len(band_fields))]
                    
                    if class_name in ref_spectra:
                        arcpy.AddWarning(f"Duplicate class name '{class_name}' found in reference table")
                        continue
                    
                    # Normalize reference spectrum
                    magnitude = math.sqrt(sum(v*v for v in band_values))
                    if magnitude > 0:
                        normalized = [v/magnitude for v in band_values]
                        ref_spectra[class_name] = normalized
                        class_names.append(class_name)
                    else:
                        arcpy.AddWarning(f"Reference spectrum for '{class_name}' has zero magnitude and will be ignored")
            
            arcpy.AddMessage(f"Found {len(ref_spectra)} reference spectra:")
            for name in class_names:
                arcpy.AddMessage(f"  - {name}")
            
            if not ref_spectra:
                arcpy.AddError("No valid reference spectra found in the reference table")
                return
            
            # Create output paths
            out_class_path = os.path.join(out_workspace, out_raster)
            out_sam_path = os.path.join(out_workspace, out_sam) if out_sam else None
            
            # Perform SAM calculation
            arcpy.AddMessage("Performing SAM classification...")
            
            # Create lists for map algebra expressions
            band_expressions = [f"Float('{bands[i+1]}')" for i in range(len(band_fields))]
            
            # Create SAM rasters for each reference spectrum
            sam_rasters = {}
            
            # 1. Compute normalization factor for pixel vectors
            norm_expr = " + ".join([f"Power({expr}, 2)" for expr in band_expressions])
            norm_raster = arcpy.sa.SquareRoot(arcpy.sa.Raster(arcpy.sa.Float(norm_expr)))
            
            # 2. Compute SAM for each reference spectrum
            for class_name, ref_vector in ref_spectra.items():
                # Calculate dot product
                dot_expr = " + ".join([f"{expr} * {ref_vector[i]}" for i, expr in enumerate(band_expressions)])
                dot_raster = arcpy.sa.Float(dot_expr)
                
                # Calculate SAM angle (arccos of dot product divided by magnitudes)
                # Since reference vectors are already normalized, we only need to normalize the pixel vector
                sam_angle = arcpy.sa.ACos(dot_raster / norm_raster)
                
                # Convert from radians to degrees
                sam_angle_deg = sam_angle * (180.0 / math.pi)
                
                # Store the SAM raster
                sam_rasters[class_name] = sam_angle_deg
            
            # 3. Create classification raster
            arcpy.AddMessage("Creating final classification raster...")
            
            # Initialize with NoData
            class_raster = arcpy.sa.SetNull(norm_raster > 0, 0)
            min_angle_raster = arcpy.sa.SetNull(norm_raster > 0, 90)  # Initialize with 90 degrees
            
            # Loop through classes to find minimum angle
            for idx, class_name in enumerate(class_names, 1):
                sam_raster = sam_rasters[class_name]
                
                # Update classification where this class has smaller angle
                class_raster = arcpy.sa.Con(
                    arcpy.sa.BooleanAnd(
                        sam_raster < min_angle_raster,
                        sam_raster <= max_angle_rad * (180.0 / math.pi)
                    ),
                    idx,
                    class_raster
                )
                
                # Update minimum angle raster
                min_angle_raster = arcpy.sa.Con(
                    sam_raster < min_angle_raster,
                    sam_raster,
                    min_angle_raster
                )
            
            # Save output classification raster
            arcpy.AddMessage(f"Saving classification raster to: {out_class_path}")
            class_raster.save(out_class_path)
            
            # Save SAM angle raster if requested
            if out_sam_path:
                arcpy.AddMessage(f"Saving SAM angle raster to: {out_sam_path}")
                min_angle_raster.save(out_sam_path)
            
            # Apply color map if requested
            if apply_color:
                arcpy.AddMessage("Applying color map to classification raster...")
                
                # Create color map
                color_map = []
                for idx, class_name in enumerate(class_names, 1):
                    # Generate a color based on index
                    hue = (idx * 137) % 360  # Use golden ratio to distribute colors
                    rgb = self._hsv_to_rgb(hue/360.0, 0.8, 0.9)
                    color_map.append([idx, class_name, rgb[0], rgb[1], rgb[2]])
                
                # Apply color map
                try:
                    arcpy.AddMessage("Setting classification symbology...")
                    result = arcpy.management.ApplySymbologyFromLayer(out_class_path, "")
                    
                    # Get the output raster layer
                    layer = result.getOutput(0)
                    
                    # Apply custom color map
                    for entry in color_map:
                        idx, name, r, g, b = entry
                        # Add code to apply color to layer
                        arcpy.AddMessage(f"  Class {idx}: {name} - RGB({r},{g},{b})")
                except Exception as e:
                    arcpy.AddWarning(f"Could not apply color map: {str(e)}")
            
            # Return output paths
            return out_class_path
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with table: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_training(self, input_raster, training_samples, max_angle_rad, threshold,
                          out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using training samples/ROIs"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Training samples: {training_samples}")
            
            # Look for class field in training samples
            class_field = None
            desc = arcpy.Describe(training_samples)
            
            # Check possible field names
            possible_fields = ["class", "classname", "class_name", "category", "label"]
            for field in desc.fields:
                if field.type == "String" and field.name.lower() in possible_fields:
                    class_field = field.name
                    break
            
            if not class_field:
                arcpy.AddWarning("No suitable class field found in training samples")
                arcpy.AddWarning("Using first string field as class field")
                
                # Use first string field
                for field in desc.fields:
                    if field.type == "String":
                        class_field = field.name
                        break
            
            if not class_field:
                arcpy.AddError("No string field found in training samples for classification")
                return None
            
            arcpy.AddMessage(f"Using field '{class_field}' for class names")
            
            # Extract values to points
            arcpy.AddMessage("Extracting spectral values from training samples...")
            
            # Create a temporary table to store extracted values
            temp_table = arcpy.CreateUniqueName("sam_training", arcpy.env.scratchGDB)
            arcpy.sa.ExtractValuesToTable(
                training_samples,
                input_raster,
                temp_table,
                "NONE",
                "ALL"
            )
            
            # Add class field to the temporary table
            arcpy.management.AddJoin(
                temp_table,
                "OBJECTID",
                training_samples,
                "OBJECTID",
                "KEEP_ALL"
            )
            
            # Get class names and band fields
            fields = [f.name for f in arcpy.ListFields(temp_table)]
            training_class_field = f"{os.path.basename(training_samples)}.{class_field}"
            
            # Get band fields (BAND_1, BAND_2, etc.)
            band_fields = [f for f in fields if f.startswith("BAND_")]
            
            # Sort band fields numerically
            band_fields.sort(key=lambda x: int(x.split("_")[1]))
            
            # Now use the table-based SAM with our extracted training data
            self._sam_with_table(
                input_raster=input_raster,
                ref_table=temp_table,
                class_field=training_class_field,
                band_fields=band_fields,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_table)
            except:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with training samples: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_endmembers(self, input_raster, endmember_pixels, max_angle_rad, threshold,
                            out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using endmember pixels"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage("Using endmember pixels for reference spectra")
            
            # Create temporary points feature class
            temp_points = arcpy.CreateUniqueName("endmember_points", arcpy.env.scratchGDB)
            
            # Create points feature class
            arcpy.management.CreateFeatureclass(
                os.path.dirname(temp_points),
                os.path.basename(temp_points),
                "POINT",
                spatial_reference=arcpy.Describe(input_raster).spatialReference
            )
            
            # Add CLASS field
            arcpy.management.AddField(temp_points, "CLASS", "TEXT")
            
            # Add points
            with arcpy.da.InsertCursor(temp_points, ["SHAPE@XY", "CLASS"]) as cursor:
                for row in endmember_pixels:
                    class_name = row[0]
                    x = row[1]
                    y = row[2]
                    cursor.insertRow([(x, y), class_name])
            
            # Now use the training samples approach with our temporary points
            self._sam_with_training(
                input_raster=input_raster,
                training_samples=temp_points,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_points)
            except:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with endmember pixels: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        if s == 0.0:
            return (int(v * 255), int(v * 255), int(v * 255))
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        if i == 0:
            return (int(v * 255), int(t * 255), int(p * 255))
        elif i == 1:
            return (int(q * 255), int(v * 255), int(p * 255))
        elif i == 2:
            return (int(p * 255), int(v * 255), int(t * 255))
        elif i == 3:
            return (int(p * 255), int(q * 255), int(v * 255))
        elif i == 4:
            return (int(t * 255), int(p * 255), int(v * 255))
        else:
            return (int(v * 255), int(p * 255), int(q * 255))