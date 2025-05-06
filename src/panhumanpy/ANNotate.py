"""
Azimuth Cell Annotation: Neural network-based hierarchical cell type 
annotation for single-cell RNA-seq data.

This module provides tools for hierarchical cell type annotation and 
interpretation based on single-cell RNA-seq data using the Azimuth 
neural network model trained on annotated panhuman scRNA-seq data.

Key Components
-------------
AzimuthNN_base : Class
    Low-level class providing fine-grained control over the annotation 
    process. Suitable for advanced users who need detailed control or 
    are processing data in batches to optimize memory usage.

AzimuthNN : Class
    High-level interface that wraps around AzimuthNN_base for interactive 
    analysis. Provides a streamlined workflow for cell annotation with 
    sensible defaults. Recommended for most interactive analysis sessions 
    and notebooks.

annotate_core : Function
    Core function for script-based automated annotation. Designed for 
    batch processing and integration into analysis pipelines.

Usage Examples
-------------
Interactive usage with high-level interface:
    >>> import anndata
    >>> from panhumanpy import AzimuthNN
    >>> adata = anndata.read_h5ad('my_data.h5ad')
    >>> azimuth = AzimuthNN(adata) # Run minimal annotation pipeline
    >>> azimuth.azimuth_refine()  # Refine annotations
    >>> embeddings = azimuth.azimuth_embed()  # Extract embeddings
    >>> umap = azimuth.azimuth_umap()  # Generate UMAP
    >>> adata_annotated = azimuth.pack_adata('output.h5ad')  # Save results

For more detailed documentation on specific classes and functions:
    >>> help(AzimuthNN)
    >>> help(AzimuthNN_base)
    >>> help(annotate_core)

Command-line Usage
-----------------
This module can be run as a standalone script to annotate h5ad files:

    annotate /path/to/input.h5ad [options]

Required positional argument:
    filepath               Path to input h5ad file containing 
                            single-cell data

Optional arguments:
    -fn, --feature_names_col
                           Column in query.var containing gene 
                            names (default: None)
    -ap, --annotation_pipeline
                           Annotation pipeline to use 
                           (default: 'supervised')
    -ebs, --eval_batch_size
                           Batch size for model inference 
                           (default: 8192)
    -norm, --normalization_override
                           Skip normalization check (default: False)
    -ncbs, --norm_check_batch_size
                           Number of cells to sample for normalization 
                           check (default: 100)
    -om, --output_mode     Output verbosity, 'minimal' or 'detailed' 
                            (default: 'minimal')
    -rf, --refine_labels   Skip hierarchical label refinement
                           (default: use refinement)
    -em, --extract_embeddings
                           Skip neural network embeddings extraction
                           (default: extract embeddings)
    -umap, --umap_embeddings
                           Skip UMAP projection generation
                           (default: generate UMAP)

UMAP parameters:
    -nnbrs, --n_neighbors  Neighbors per point in UMAP (default: 30)
    -nc, --n_components    UMAP dimensionality (default: 2)
    -me, --metric          Distance metric for UMAP (default: 'cosine')
    -mdt, --min_dist       Minimum distance in UMAP (default: 0.3)
    -ulr, --umap_lr        UMAP learning rate (default: 1.0)
    -useed, --umap_seed    Random seed for reproducibility (default: 42)
    -sp, --spread          UMAP spread parameter (default: 1.0)
    -uv, --umap_verbose    Hide UMAP progress
                           (default: show progress)
    -uin, --umap_init      UMAP initialization method (default: 'spectral')

Output:
The annotated data will be saved as a new h5ad file in the same directory
as the input file, with '_ANN' appended to the filename. If a file with 
that name already exists, a timestamp (YYYYMMDD_HHMMSS) will be 
automatically appended to prevent overwriting existing results.

Example command:
    annotate my_cells.h5ad -fn feature_name -ebs 4096 -nc 3
"""



from panhumanpy.ANNotate_tools import *

_gpu_configured = False

def configure_once():
    """
    Configures TensorFlow GPU settings once per process.
    This wrapper ensures the configuration only happens once.
    """
    global _gpu_configured
    if not _gpu_configured:
        configure()
        _gpu_configured = True
        return True
    return False

configure_once()


########################################################################
###### Base class for low level interactive usage ######################


class AzimuthNN_base(AutoloadInferenceTools):
    """
    Base class for low-level interactive usage of the Azimuth neural 
    network annotation pipeline.
    
    This class provides a comprehensive framework for single-cell 
    RNA-seq annotation using neural network models. It handles the 
    complete workflow from data loading and preprocessing to inference, 
    post-processing, and result visualization. This includes functionality 
    for extracting embeddings, generating UMAP visualizations, and refining
    annotations at different levels of granularity.
    
    Parameters
    ----------
    annotation_pipeline : str, default='supervised'
        The type of annotation pipeline to use.
    eval_batch_size : int, default=8192
        Batch size for inference and embedding generation.
        
    Attributes
    ----------
    query : anndata.AnnData or None
        The AnnData object if loaded.
    X_query : scipy.sparse.csr_matrix or None
        Expression matrix in CSR format.
    query_features : list or None
        List of feature names.
    features_meta : pandas.DataFrame or None
        Feature metadata.
    cells_meta : pandas.DataFrame or None
        Cell metadata.
    num_cells : int or None
        Number of cells in the query.
    processed_outputs : dict or None
        Processed inference results.
    embeddings : dict
        Dictionary of extracted embeddings.
    umaps : dict
        Dictionary of generated UMAP coordinates.
        
    Raises
    ------
    TypeError
        If input parameters are not of the correct type.
    RuntimeError
        If model metadata fails to load.
        
    Notes
    -----
    This class is designed for programmatic use and provides 
    fine-grained control over each step of the annotation pipeline. 
    Consider using a higher-level interface for convenience if a 
    standard workflow is sufficient for your needs.
    """
    def __init__(
        self, 
        annotation_pipeline='supervised',
        eval_batch_size=8192
        ):

        if not isinstance(annotation_pipeline, str):
            raise TypeError("annotation_pipeline must be a string")
            
        if not isinstance(eval_batch_size, int):
            raise TypeError("eval_batch_size must be an integer")

        

        self._annotation_pipeline = annotation_pipeline
        self._eval_batch_size = eval_batch_size

       
        super().__init__(annotation_pipeline)

        if not hasattr(self, 'model_meta'):
            raise RuntimeError("Failed to load model metadata")
            
        for meta_key in self.model_meta.keys():
            if not isinstance(meta_key, str):
                raise TypeError("All model metadata keys must be strings")
            setattr(self, meta_key, self.model_meta[meta_key])

        self.query = None
        self.X_query = None
        self.query_features = None
        self.features_meta = None
        self.cells_meta = None
        self.num_cells = None
        self._inference_input_matrix = None

        self._inference_outputs_unprocessed = None
        self.processed_outputs = None
        self._azimuth_refined_labels = {}
        self.embeddings = {}
        self.umaps = {}

        

    def query_stripped(
        self,
        X_query,
        query_features,
        cells_meta
        ):
        """
        Load query data directly from expression matrix and metadata.
        
        This method allows for direct loading of pre-processed 
        expression data without requiring an AnnData object. This is 
        useful for integration with custom preprocessing pipelines.
        
        Parameters
        ----------
        X_query : scipy.sparse.csr_matrix
            Expression matrix with cells as rows and features as columns.
        query_features : list of str
            List of feature names corresponding to columns in X_query.
        cells_meta : pandas.DataFrame
            Cell metadata with rows corresponding to cells in X_query.
            
        Raises
        ------
        TypeError
            If inputs are not of correct type.
        ValueError
            If dimensions of inputs don't match.
            
        Notes
        -----
        This method creates a minimal features_meta DataFrame based on 
        the provided feature names.
        """
        if not isinstance(X_query, csr_matrix):
            raise TypeError("X_query must be a scipy.sparse.csr_matrix")
            
        if not isinstance(query_features, list) or not all(isinstance(f, str) 
            for f in query_features):
            raise TypeError("query_features must be a list of strings")
            
        if not isinstance(cells_meta, pd.DataFrame):
            raise TypeError("cells_meta must be a pandas DataFrame")
            
        if len(query_features) != X_query.shape[1]:
            raise ValueError(
                f"Number of features ({len(query_features)}) "
                f"does not match X_query columns ({X_query.shape[1]})"
            )
            
        if cells_meta.shape[0] != X_query.shape[0]:
            raise ValueError(
                f"Number of cells in metadata "
                f"({cells_meta.shape[0]}) does not match X_query rows "
                f"({X_query.shape[0]})"
                )

        self.X_query = X_query
        self.query_features = query_features
        self.features_meta = pd.DataFrame(
            {'feature_name':query_features},
            index = query_features
            )
        self.cells_meta = cells_meta
        self.num_cells = X_query.shape[0]


    def query_adata(
        self, 
        query_arg,
        feature_names_col=None
        ):
        """
        Load query data from an AnnData object.
        
        Parameters
        ----------
        query_arg : anndata.AnnData
            AnnData object containing expression data and metadata.
        feature_names_col : str, optional
            Column in var DataFrame to use for feature names.
            If None, uses the var_names index.
            
        Notes
        -----
        This method extracts the expression matrix, feature names,
        and metadata from the provided AnnData object.
        """
        query_obj = QueryObj(query_arg)

        self.X_query = query_obj.X_query()
        self.query_features = query_obj.query_features(
            feature_names_col=feature_names_col
        )
        self.features_meta = query_obj.features_meta()
        self.cells_meta = query_obj.cells_meta()
        self.num_cells = self.X_query.shape[0]

    def query_h5ad(
        self, 
        query_filepath,
        feature_names_col=None
        ):
        """
        Load query data from an H5AD file on disk.
        
        Parameters
        ----------
        query_filepath : str
            Path to the H5AD file containing the query data.
        feature_names_col : str, optional
            Column in var DataFrame to use for feature names.
            If None, uses the var_names index.
            
        Raises
        ------
        ValueError
            If the file is not in H5AD format.
            
        Notes
        -----
        This method reads the H5AD file from disk and extracts the
        necessary components for inference.
        """
        query_obj = ReadQueryObj(query_filepath)

        self.X_query = query_obj.X_query()
        self.query_features = query_obj.query_features(
            feature_names_col=feature_names_col
        )
        self.features_meta = query_obj.features_meta()
        self.cells_meta = query_obj.cells_meta()
        self.num_cells = self.X_query.shape[0]

    def process_query(
        self,
        normalization_override=False,
        norm_check_batch_size=100
        ):
        """
        Process the query data to prepare it for inference.
        
        This method prepares the expression data for the inference model 
        according to the specified annotation pipeline. The processing steps
        vary depending on the pipeline type, potentially including 
        normalization, feature selection, dimensionality reduction, or 
        other transformations.
        
        Parameters
        ----------
        normalization_override : bool, default=False
            If True, bypasses normalization entirely regardless of
            whether the values are integers or not.
        norm_check_batch_size : int, default=1000
            Batch size for checking normalization status.
            
        Raises
        ------
        TypeError
            If parameters are not of the correct type.
            
        Notes
        -----
        This method must be called after loading query data and before
        running inference or extracting embeddings. The specific processing
        steps depend on the annotation_pipeline specified during 
        initialization.
        
        Currently, only the 'supervised' annotation pipeline is implemented,
        which normalizes the expression data and aligns it with a reference
        feature panel.
        """

        if not isinstance(normalization_override, bool):
            raise TypeError("normalization override must be a bool")

        if not isinstance(norm_check_batch_size, int):
            raise TypeError("norm_check_batch_size must be an integer")
        
        query_processing_class = InferenceInputData(
            self.X_query,
            self.query_features,
            self.inference_feature_panel,
            normalization_override = normalization_override,
            norm_check_batch_size = norm_check_batch_size
        )

        self._inference_input_matrix = query_processing_class.inference_input(
            annotation_pipeline = self._annotation_pipeline
        )
        

    def run_inference_model(self):
        """
        Run the inference model on the processed query data.
        
        This method executes the neural network inference to generate
        cell type predictions.
        
        Returns
        -------
        dict
            Dictionary of raw inference outputs including hierarchical
            label predictions and probabilities.
            
        Raises
        ------
        AssertionError
            If input matrix has not been initialized by calling 
            process_query().
            
        Notes
        -----
        The raw outputs should typically be processed using the 
        process_outputs() method before further use downstream.
        """

        assert self._inference_input_matrix is not None, (
            "Input matrix not initialized. Call process_query() first."
        )

        inference_class = Inference(
            self._inference_input_matrix,
            self.inference_model,
            self.inference_encoders,
            self._eval_batch_size,
            self.max_depth
        )

        self._inference_outputs_unprocessed = inference_class.run_inference()

        return self._inference_outputs_unprocessed


    def process_outputs(self, mode='minimal'):
        """
        Process raw inference outputs into usable predictions.
        
        This method organizes the raw inference outputs into a structured
        dictionary of predictions at various hierarchical levels.
        
        Parameters
        ----------
        mode : str, default='minimal'
            Processing mode: 'minimal' provides essential outputs,
            'detailed' includes additional information for all levels.
            
        Returns
        -------
        dict
            Dictionary of processed outputs including hierarchical labels,
            level-specific labels, and confidence scores.
            
        Raises
        ------
        AssertionError
            If mode is not 'minimal' or 'detailed'.
            
        Notes
        -----
        This method should be called after run_inference_model().
        """

        assert mode in ['minimal','detailed'], (
            "mode for output processing should be either "
            "'minimal' or 'detailed'"
        )

        labels_pred = self._inference_outputs_unprocessed[
            'hierarchical_label_preds'
            ]
        labels_prob = self._inference_outputs_unprocessed[
            'probability_of_preds'
            ]

        output_processing_class = OutputLabels(
            labels_pred,
            labels_prob,
            self.max_depth,
            self.num_cells
        )

        combined_labels = output_processing_class.combined_labels
        level_zero_labels = output_processing_class.level_zero_labels
        final_level_labels = output_processing_class.final_level_labels
        final_level_softmax_prob = (
            output_processing_class.final_level_softmax_prob
        )
        full_consistent_hierarchy = (
            output_processing_class.full_consistent_hierarchy
        )

        self.processed_outputs = {
            'full_hierarchical_labels': combined_labels,
            'level_zero_labels': level_zero_labels,
            'final_level_labels': final_level_labels,
            'final_level_softmax_prob': final_level_softmax_prob,
            'full_consistent_hierarchy': full_consistent_hierarchy
        }

        if mode=='detailed':
            for i in range(1, self.max_depth):
                self.processed_outputs[f'level_{i}_labels'] = (
                output_processing_class.all_level_labels()[i]
            )

        return self.processed_outputs

    def refine_labels(self, refine_level):
        """
        Refine hierarchical labels to a consistent level of granularity.
        
        This method applies post-processing rules to standardize 
        annotations at the specified level of granularity (broad, 
        medium, or fine).
        
        Parameters
        ----------
        refine_level : str
            Level of refinement: 'broad', 'medium', or 'fine'.
            
        Returns
        -------
        list
            List of refined labels at the specified level.
            
        Raises
        ------
        AssertionError
            If refine_level is not valid or inference hasn't been run.
            
        Notes
        -----
        For 'broad' level, this returns the top-level annotations.
        For 'medium' and 'fine' levels, specialized refinement is 
        applied.
        """

        assert refine_level in ['broad','medium','fine'], (
            "refine_level should be 'broad', 'medium', or 'fine'."
        )

        assert self._inference_outputs_unprocessed is not None, (
            "Labels can be refined only after inference model has been run."
        )

        print(
            "Interpreting label predictions for consistent granularity "
            f"at {refine_level} level.\n")

        if refine_level in ['medium', 'fine']:

            labels_pred = self._inference_outputs_unprocessed[
                'hierarchical_label_preds'
            ]
            labels_prob = self._inference_outputs_unprocessed[
                'probability_of_preds'
            ]
            softmax_probs = self._inference_outputs_unprocessed[
                'softmax_vals_all'
            ]

            refine_class = PostprocessingAzimuthLabels(
                labels_pred,
                labels_prob,
                self.max_depth,
                self.num_cells,
                softmax_probs,
                self.inference_encoders,
                refine_level
            )

            results = refine_class.refine_labels()
        
        else:
            results = self.processed_outputs['level_zero_labels']

        (
            self._azimuth_refined_labels[f'azimuth_{refine_level}']
         ) = results

        return  results
    
    def add_refined_score(self, medium_labels=None):
        labels_pred = self.processed_outputs['full_hierarchical_labels']
        labels_prob = self._inference_outputs_unprocessed['probability_of_preds']

        if medium_labels is not None:
            probs = []
            for i, medium_label in enumerate(medium_labels):
                full_label = labels_pred[i]
                if medium_label not in [None, 'False']:
                    parts = full_label.split("|")
                    refined_level = None
                    for level in range(1, len(parts) + 1):
                        if parts[:level][-1] == medium_label:
                            refined_level = level
                            break
                    original_level = labels_pred[i].count("|") 
                    if refined_level is not None and refined_level <= original_level:
                        probs.append(labels_prob[i][refined_level-1])
                        continue
                probs.append(None)

            self.processed_outputs['azimuth_medium_prob'] = probs # Can change name of score here 

        return probs

    def inference_model_embeddings(self, embedding_layer_name):
        """
        Extract embeddings from an intermediate layer of the inference 
        model.
        
        Parameters
        ----------
        embedding_layer_name : str
            Name of the layer to extract embeddings from.
            
        Returns
        -------
        numpy.ndarray
            Embeddings from the specified layer for all query cells.
            
        Raises
        ------
        RuntimeError
            If inference model is not found.
        AssertionError
            If input matrix has not been initialized.
            
        Notes
        -----
        The embeddings are stored in the embeddings dictionary with a 
        key that combines the model name and layer name.
        """

        if not hasattr(self, 'inference_model'):
            raise RuntimeError("inference_model not found")
            
        assert self._inference_input_matrix is not None, (
            "Input matrix not initialized. Call process_query() first."
        )

        embedding_class = Embeddings(self.inference_model, embedding_layer_name)

        embeddings = embedding_class.embeddings(
            self._inference_input_matrix,
            self._eval_batch_size
            )

        self.embeddings[
            f'{self.inference_model_name}_{embedding_layer_name}_embed'
            ] = embeddings

        return embeddings


    def inference_model_umaps(
        self,
        embedding_layer_name,
        n_neighbors=30,
        n_components=2,
        metric='cosine',
        min_dist=0.3,
        umap_lr=1.0,
        umap_seed=42,
        spread=1.0,
        verbose=True,
        init='spectral'
        ):
        """
        Generate UMAP coordinates from existing embeddings.
        
        Parameters
        ----------
        embedding_layer_name : str
            Name of the layer whose embeddings should be used.
        n_neighbors : int, default=30
            Number of neighbors for UMAP.
        n_components : int, default=2
            Number of dimensions for UMAP output.
        metric : str, default='cosine'
            Distance metric for UMAP.
        min_dist : float, default=0.3
            Minimum distance parameter for UMAP.
        umap_lr : float, default=1.0
            Learning rate for UMAP.
        umap_seed : int, default=42
            Random seed for reproducibility.
        spread : float, default=1.0
            Spread parameter for UMAP.
        verbose : bool, default=True
            Whether to display progress during UMAP calculation.
        init : str, default='spectral'
            Initialization method for UMAP.
            
        Returns
        -------
        numpy.ndarray
            UMAP coordinates for all query cells.
            
        Raises
        ------
        AssertionError
            If the specified embeddings have not been generated.
            
        Notes
        -----
        This method requires that embeddings have already been generated
        using inference_model_embeddings().
        """

        embedding_key_base = (
            f"{self.inference_model_name}_{embedding_layer_name}"
        )
        embedding_key = embedding_key_base + '_embed'
        umap_key = embedding_key_base + '_umap'

        assert embedding_key in self.embeddings.keys(), (
            f"Embedding '{embedding_key}' not found. Generate embeddings "
            "before creating umap."
        )

        umap_class = Umaps(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            umap_lr=umap_lr,
            umap_seed=umap_seed,
            spread=spread,
            verbose=verbose,
            init=init
        )

        umap_gen = umap_class.create_umap(self.embeddings[embedding_key])
        self.umaps[umap_key] = umap_gen

        return umap_gen

    def inference_embeddings_and_umap(
        self, 
        embedding_layer_name,        
        n_neighbors=30,
        n_components=2,
        metric='cosine',
        min_dist=0.3,
        umap_lr=1.0,
        umap_seed=42,
        spread=1.0,
        verbose=True,
        init='spectral'
        ):
        """
        Generate both embeddings and UMAP coordinates in one operation.
        
        This is a convenience method that combines the functionality of
        inference_model_embeddings() and inference_model_umaps().
        
        Parameters
        ----------
        embedding_layer_name : str
            Name of the layer to extract embeddings from.
        n_neighbors : int, default=30
            Number of neighbors for UMAP.
        n_components : int, default=2
            Number of dimensions for UMAP output.
        metric : str, default='cosine'
            Distance metric for UMAP.
        min_dist : float, default=0.3
            Minimum distance parameter for UMAP.
        umap_lr : float, default=1.0
            Learning rate for UMAP.
        umap_seed : int, default=42
            Random seed for reproducibility.
        spread : float, default=1.0
            Spread parameter for UMAP.
        verbose : bool, default=True
            Whether to display progress during UMAP calculation.
        init : str, default='spectral'
            Initialization method for UMAP.
            
        Returns
        -------
        tuple
            Tuple containing (embeddings, umap_coordinates).
            
        Raises
        ------
        RuntimeError
            If inference model is not found.
        ValueError
            If input matrix has not been initialized.
            
        Notes
        -----
        This method may be more efficient than calling the two component
        methods separately, depending on usage, as it avoids storing 
        intermediate results in memory twice.
        """

        if not hasattr(self, 'inference_model'):
            raise RuntimeError("inference_model not found")
            
        if self._inference_input_matrix is None:
            raise ValueError(
                "X_query has not been processed for extraction of embeddings.\n"
                "Run process_query() first."
            )

        embeddings_and_umap_class = EmbeddingsAndUmap(
            self.inference_model,
            embedding_layer_name,
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            umap_lr=umap_lr,
            umap_seed=umap_seed,
            spread=spread,
            verbose=verbose,
            init=init
        )

        em, umap_em = embeddings_and_umap_class.create_embeddings_and_umap(
            self._inference_input_matrix,
            self._eval_batch_size
        )

        embed_key = f'{self.inference_model_name}_{embedding_layer_name}_embed'
        umap_key = f'{self.inference_model_name}_{embedding_layer_name}_umap'
        
        self.embeddings[embed_key] = em
        self.umaps[umap_key] = umap_em

        return em, umap_em

    def update_cells_meta(self):
        """
        Update cells_meta DataFrame with inference results.
        
        This method adds or updates columns in the cell metadata DataFrame
        with inference results, including predictions and refined labels.
        
        Returns
        -------
        pandas.DataFrame
            Updated cell metadata DataFrame.
            
        Raises
        ------
        TypeError
            If cells_meta is not a pandas DataFrame.
        RuntimeError
            If num_cells attribute is not set.
        ValueError
            If the number of values doesn't match the number of cells.
            
        Notes
        -----
        If both broad level annotations and level_zero_labels exist,
        the latter is dropped to avoid duplication.
        """

        if not isinstance(self.cells_meta, pd.DataFrame):
            raise TypeError(
                "Query cell meta is not available as pandas dataframe"
                )
            
        if not hasattr(self, 'num_cells'):
            raise RuntimeError("num_cells not found")

        if self.processed_outputs and len(self.processed_outputs) > 0:
            for col_idx, (meta_col, values) in enumerate(
                self.processed_outputs.items()
            ):
                if len(values) != self.num_cells:
                    raise ValueError(
                        f"Column {meta_col} has {len(values)} values but "
                        f"expected {self.num_cells}"
                    )
                self.cells_meta = insert_col(
                    self.cells_meta, 
                    col_idx,
                    meta_col,
                    values
                )

        if (
            len(self._azimuth_refined_labels)
         ) > 0:
            start_idx = len(self.processed_outputs or [])
            for col_idx, (refine_col, values) in enumerate(
                self._azimuth_refined_labels.items(), start=start_idx
            ):
                if len(values) != self.num_cells:
                    raise ValueError(
                        f"Column {refine_col} has {len(values)} values but "
                        f"expected {self.num_cells}"
                    )
                self.cells_meta = insert_col(
                    self.cells_meta,
                    col_idx,
                    refine_col,
                    values
                )

            if (
                (
                    "azimuth_broad" in self.cells_meta.keys()
                ) and (
                    "level_zero_labels" in self.cells_meta.keys()
                    )
                ):
                self.cells_meta.drop('level_zero_labels', axis=1, inplace=True)


        return self.cells_meta


    def pack_adata(self, save_path = None):
        """
        Create an AnnData object with all results and optionally save to
         disk.
        
        This method packages all results (expression data, metadata, 
        embeddings, and UMAP coordinates) into a unified AnnData object 
        for further analysis or visualization.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the AnnData object as an H5AD file.
            If None, the object is created but not saved.
            
        Returns
        -------
        anndata.AnnData
            AnnData object containing all query data and results.
            
        Notes
        -----
        If the specified save_path already exists, a timestamp is 
        appended to the filename to prevent overwriting.
        """

        all_embeddings = {**self.embeddings, **self.umaps}

        adata_obj = create_anndata(
            self.X_query,
            self.cells_meta,
            self.features_meta,
            embeddings = all_embeddings
        )

        if save_path:
            if os.path.exists(save_path):
                
                base_path, ext = os.path.splitext(save_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{base_path}_{timestamp}{ext}"
                print(
                    f"File {save_path} already exists. Adding timestamp "
                    "suffix to prevent overwrite.",
                )
                print(f"Saving to: {save_path}")
            adata_obj.write(save_path)

        self.query = adata_obj

        return adata_obj




########################################################################
######## Object for high level interactive usage #######################

class AzimuthNN(AzimuthNN_base):
    """
    AzimuthNN: A high-level interface for a cell annotation pipeline 
    based on the Azimuth neural network.
    
    This class wraps around the AzimuthNN_base class to provide a 
    simplified workflow for hierarchical cell type annotation based on 
    single-cell RNA-seq data, handling data loading, preprocessing, 
    model inference, and visualization in a streamlined manner. 
    
    For more fine-grained control over the annotation process, users 
    should directly use the AzimuthNN_base class.
    
    Parameters
    ----------
    query_arg : Union[str, anndata.AnnData]
        Either an AnnData object containing single-cell data or a path 
        to an h5ad file.
    feature_names_col : str, optional
        Column in the anndata_object.var dataframe that contains the 
        gene names to use for model input. If None, assumes var_names 
        are already the correct gene identifiers.
    annotation_pipeline : str, default='supervised'
        Type of annotation pipeline to use for cell type prediction.
    eval_batch_size : int, default=8192
        Batch size to use during model inference.
    normalization_override : bool, default=False
        If True, skips normalization check and forces processing to 
        continue.
    norm_check_batch_size : int, default=1000
        Number of cells to sample for normalization check.
    output_mode : str, default='minimal'
        Controls the verbosity of output in the cell meta dataframe. 
        Options are 'minimal' or 'detailed'.
        
    Attributes
    ----------
    cells_meta : pandas.DataFrame or None
        Cell metadata.
    embeddings : dict
        Contains embeddings extracted from the model.
    umaps : dict
        Contains UMAP projections of the embeddings.
        
    Raises
    ------
    TypeError
        If query_arg is not an AnnData object or a string path,
        if normalization_override is not a bool,
        or if norm_check_batch_size is not an integer.
    ValueError
        If output_mode is not 'minimal' or 'detailed'.
    
    Examples
    --------
    >>> import anndata
    >>> adata = anndata.read_h5ad('my_data.h5ad')
    >>> azimuth = AzimuthNN(adata)
    >>> azimuth.azimuth_refine()
    >>> embeddings = azimuth.azimuth_embed()
    >>> umap = azimuth.azimuth_umap()
    >>> cell_metadata = azimuth.cells_meta
    """

    def __init__(
        self,
        query_arg,
        feature_names_col=None,
        annotation_pipeline='supervised',
        eval_batch_size=8192,
        normalization_override=False,
        norm_check_batch_size=100,
        output_mode='minimal'
        ):

        if (
            not isinstance(query_arg, str) and 
            not isinstance(query_arg, anndata.AnnData)
            ):
            raise TypeError(
                "query argument must either be AnnData object or "
                "path to an h5ad file."
                )

        if feature_names_col is None:
            warnings.warn(
                "Ensure that the features metadata is indexed with gene names.",
                UserWarning
                )

        if not isinstance(normalization_override, bool):
            raise TypeError("normalization override must be a bool")

        if not isinstance(norm_check_batch_size, int):
            raise TypeError("norm_check_batch_size must be an integer")

        if output_mode not in ['minimal', 'detailed']:
            raise ValueError(
                "output_mode must be either 'minimal' or 'detailed'"
            )

        self._query_arg = query_arg
        self._annotation_pipeline = annotation_pipeline
        self._eval_batch_size = eval_batch_size
        self._normalization_override = normalization_override
        self._norm_check_batch_size = norm_check_batch_size
        self._output_mode = output_mode

        super().__init__(
            self._annotation_pipeline,
            self._eval_batch_size
        )

        if isinstance(self._query_arg, anndata.AnnData):
            self.query_adata(
                self._query_arg,
                feature_names_col = feature_names_col
                )
        elif isinstance(self._query_arg, str):
            self.query_h5ad(
                self._query_arg,
                feature_names_col = feature_names_col
                )

        self.process_query(
            normalization_override = self._normalization_override,
            norm_check_batch_size = self._norm_check_batch_size
        )

        _ = self.run_inference_model()
        self.annotations = self.process_outputs(mode=self._output_mode)
        _ = self.update_cells_meta()

    def azimuth_refine(self):
        """
        Refine cell type annotations at multiple granularity levels.
        
        This method applies a hierarchical refinement of cell type labels,
        progressing from broad to fine classifications. The results are
        stored in the annotations attribute and the cell metadata is 
        updated. The softmax probability values for the medium level are 
        also added to the cell metadata whenever applicable. 
        
        Returns
        -------
        None
            Updates the annotations attribute in-place and updates cell 
            metadata.
        """

        _ = self.refine_labels(refine_level='broad')
        medium_labels = self.refine_labels(refine_level='medium')
        _ = self.refine_labels(refine_level='fine')
        _ = self.add_refined_score(medium_labels)

        _ = self.update_cells_meta()

    def azimuth_embed(self):
        """
        Extract embeddings from the Azimuth model's embedding layer.
        
        This method extracts cell embeddings from a pre-defined layer in
          the inference model and stores them in the embeddings 
          dictionary under the key 'azimuth_embed', replacing the 
          original model-specific key.

        To extract embeddings from a different layer in the model, use
        AzimuthNN_base class for more fine grained control. 
        
        Returns
        -------
        numpy.ndarray
            The extracted embeddings, with shape 
            (n_cells, embedding_dimension).
            
        Raises
        ------
        AssertionError
            If inference model hasn't been run yet.
        """

        azimuth_embedding_layer_name = self.model_meta[
                'inference_model_embedding_layer'
        ]

        azimuth_embeddings = self.inference_model_embeddings(
            embedding_layer_name = azimuth_embedding_layer_name
        )

        self.embeddings['azimuth_embed'] = azimuth_embeddings
        del self.embeddings[
            f'{self.inference_model_name}_{azimuth_embedding_layer_name}_embed'
            ]

        return azimuth_embeddings

    def azimuth_umap(
        self,
        n_neighbors=30,
        n_components=2,
        metric='cosine',
        min_dist=0.3,
        umap_lr=1.0,
        umap_seed=42,
        spread=1.0,
        verbose=True,
        init='spectral'
        ):
        """
        Generate UMAP projection from Azimuth embeddings.
        
        This method creates a UMAP projection from previously extracted
        Azimuth embeddings and stores it in the umaps dictionary.
        
        Parameters
        ----------
        n_neighbors : int, default=30
            Number of neighbors to consider for each point in UMAP.
        n_components : int, default=2
            Dimensionality of the UMAP projection.
        metric : str, default='cosine'
            Distance metric to use for UMAP.
        min_dist : float, default=0.3
            Minimum distance between points in the UMAP projection.
        umap_lr : float, default=1.0
            UMAP learning rate.
        umap_seed : int, default=42
            Random seed for UMAP for reproducibility.
        spread : float, default=1.0
            Scales the effective scale of embedded points.
        verbose : bool, default=True
            Whether to display progress during UMAP computation.
        init : str, default='spectral'
            Initialization method for UMAP.
            
        Returns
        -------
        numpy.ndarray
            The UMAP projection, with shape (n_cells, n_components).
            
        Raises
        ------
        AssertionError
            If 'azimuth_embed' embeddings haven't been generated yet.
        """

        assert 'azimuth_embed' in self.embeddings.keys(), (
            "Extract azimuth_embed first before creating umap."
        )

        umap_class = Umaps(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            umap_lr=umap_lr,
            umap_seed=umap_seed,
            spread=spread,
            verbose=verbose,
            init=init
        )

        umap_gen = umap_class.create_umap(self.embeddings['azimuth_embed'])
        self.umaps['azimuth_umap'] = umap_gen

        return umap_gen

    def azimuth_embed_and_umap(
        self,
        n_neighbors=30,
        n_components=2,
        metric='cosine',
        min_dist=0.3,
        umap_lr=1.0,
        umap_seed=42,
        spread=1.0,
        verbose=True,
        init='spectral'
        ):
        """
        Extract embeddings and generate UMAP projection in one step.
        
        This method provides a convenient wrapper that combines the 
        functionality of azimuth_embed() and azimuth_umap() methods.
        It extracts embeddings from the inference model and immediately
        computes a UMAP projection, storing both results.
        
        Parameters
        ----------
        n_neighbors : int, default=30
            Number of neighbors to consider for each point in UMAP.
        n_components : int, default=2
            Dimensionality of the UMAP projection.
        metric : str, default='cosine'
            Distance metric to use for UMAP.
        min_dist : float, default=0.3
            Minimum distance between points in the UMAP projection.
        umap_lr : float, default=1.0
            UMAP learning rate.
        umap_seed : int, default=42
            Random seed for UMAP for reproducibility.
        spread : float, default=1.0
            Scales the effective scale of embedded points.
        verbose : bool, default=True
            Whether to display progress during UMAP computation.
        init : str, default='spectral'
            Initialization method for UMAP.
            
        Returns
        -------
        tuple
            A tuple containing:
                - numpy.ndarray: The extracted embeddings
                - numpy.ndarray: The UMAP projection
                
        Raises
        ------
        AssertionError
            If inference model hasn't been run yet.
        """

        azimuth_embedding_layer_name = self.model_meta[
                'inference_model_embedding_layer'
        ]

        azimuth_embeddings, azimuth_umap = self.inference_embeddings_and_umap(
            embedding_layer_name = azimuth_embedding_layer_name,
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            umap_lr=umap_lr,
            umap_seed=umap_seed,
            spread=spread,
            verbose=verbose,
            init=init
        )

        embed_key_og = (
            f'{self.inference_model_name}_{azimuth_embedding_layer_name}_embed'
        )
        umap_key_og = (
            f'{self.inference_model_name}_{azimuth_embedding_layer_name}_umap'
        )

        self.embeddings['azimuth_embed'] = azimuth_embeddings
        self.umaps['azimuth_umap'] = azimuth_umap

        del self.embeddings[embed_key_og]
        del self.umaps[umap_key_og]

        return azimuth_embeddings, azimuth_umap




################################################################################
########### functions for python and R script ##################################
################################################################################
################################################################################





########### annotate_core, core of python/R scripts ############################



def annotate_core(
    X_query,
    query_features,
    cells_meta,
    annotation_pipeline='supervised',
    eval_batch_size=8192,
    normalization_override=False,
    norm_check_batch_size=1000,
    output_mode='minimal',
    refine_labels=True,
    extract_embeddings=True,
    umap_embeddings=True,
    n_neighbors=30, 
    n_components=2, 
    metric='cosine', 
    min_dist=0.3, 
    umap_lr=1.0, 
    umap_seed=42, 
    spread=1.0,
    verbose=True,
    init='spectral'
    ):
    """
    Core function for cell type annotation using the Azimuth neural 
    network, designed primarily for script-based usage.
    
    While AzimuthNN and AzimuthNN_base classes provide interactive 
    functionality for exploratory analysis, this function offers a 
    one-step method for automated annotation via Python or R scripts. It
      performs the complete annotation workflow in a single function 
      call: data preprocessing, model inference, label generation, 
    optional label refinement, and optional embedding/UMAP generation.
    
    Parameters
    ----------
    X_query : scipy.sparse.csr_matrix
        Expression matrix with cells as rows and genes as columns.
    query_features : list of str
        List of feature names (gene identifiers) corresponding to 
        columns in X_query.
    cells_meta : pandas.DataFrame
        Metadata for cells, with rows corresponding to cells in X_query.
    annotation_pipeline : str, default='supervised'
        Type of annotation pipeline to use for cell type prediction.
    eval_batch_size : int, default=8192
        Batch size to use during model inference.
    normalization_override : bool, default=False
        If True, skips normalization check and forces processing to 
        continue.
    norm_check_batch_size : int, default=1000
        Number of cells to sample for normalization check.
    output_mode : str, default='minimal'
        Controls the verbosity of output in the cell meta dataframe.
        Options are 'minimal' or 'detailed'.
    refine_labels : bool, default=True
        Whether to perform label refinement at broad, medium, and fine 
        levels.
    extract_embeddings : bool, default=True
        Whether to extract embeddings from the model.
    umap_embeddings : bool, default=True
        Whether to generate UMAP projections from the embeddings.
        Requires extract_embeddings=True.
    n_neighbors : int, default=30
        Number of neighbors to consider for each point in UMAP.
    n_components : int, default=2
        Dimensionality of the UMAP projection.
    metric : str, default='cosine'
        Distance metric to use for UMAP.
    min_dist : float, default=0.3
        Minimum distance between points in the UMAP projection.
    umap_lr : float, default=1.0
        UMAP learning rate.
    umap_seed : int, default=42
        Random seed for UMAP for reproducibility.
    spread : float, default=1.0
        Scales the effective scale of embedded points.
    verbose : bool, default=True
        Whether to display progress during UMAP computation.
    init : str, default='spectral'
        Initialization method for UMAP.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'azimuth_object': The instantiated AzimuthNN_base object
        - 'embeddings_dict': Dictionary of computed embeddings
        - 'umap_dict': Dictionary of computed UMAP projections
        - 'cells_meta': Updated cell metadata with annotations
        
    Raises
    ------
    TypeError
        If normalization_override, extract_embeddings, umap_embeddings,
        or refine_labels are not boolean values, or if 
        norm_check_batch_size is not an integer.
    ValueError
        If output_mode is not 'minimal' or 'detailed', or if 
        umap_embeddings is True but extract_embeddings is False.
        
    Notes
    -----
    This function is designed to be the core engine for script-based
    automated annotation workflows. Unlike the interactive AzimuthNN and
    AzimuthNN_base classes which allow step-by-step exploration and 
    visualization, this function executes the entire annotation pipeline
      in one call.
    
    It's particularly useful for:
    - Batch processing of multiple datasets
    - Integration into automated analysis pipelines
    - Creating wrappers for other languages (like R)
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Example data (normally this would be your scRNA-seq data)
    >>> X = csr_matrix(np.random.rand(100, 1000))
    >>> features = [f"gene_{i}" for i in range(1000)]
    >>> meta = pd.DataFrame(index=range(100))
    >>> 
    >>> # Run annotation
    >>> results = annotate_core(X, features, meta)
    >>> 
    >>> # Access results
    >>> annotated_meta = results['cells_meta']
    >>> embeddings = results['embeddings_dict']['azimuth_embed']
    >>> umap_coords = results['umap_dict']['azimuth_umap']
    """

    if not isinstance(normalization_override, bool):
            raise TypeError("normalization override must be a bool")

    if not isinstance(norm_check_batch_size, int):
        raise TypeError("norm_check_batch_size must be an integer")

    if output_mode not in ['minimal','detailed']:
        raise ValueError(
            "mode for output processing should be either "
            "'minimal' or 'detailed'"
        )

    if not isinstance(extract_embeddings,bool):
        raise TypeError("extract_embeddings argument should be boolean")

    if not isinstance(umap_embeddings,bool):
        raise TypeError("umap_embeddings argument should be boolean")

    if not isinstance(refine_labels, bool):
        raise TypeError("refine_labels argument should be boolean") 

    if umap_embeddings:
        if not extract_embeddings:
            raise ValueError(
                "Embeddings must be extracted to create umap.\n"
                "Set extract_embeddings to True."
            )

    azimuth = AzimuthNN_base(
        annotation_pipeline = annotation_pipeline,
        eval_batch_size = eval_batch_size
    )

    azimuth.query_stripped(
        X_query,
        query_features,
        cells_meta
    )

    print("Reference model and parameters:")
    print(f"    Model name: {azimuth.model_meta['inference_model_name']}")
    print(f"    Evaluation batch size: {eval_batch_size}")
    print(f"    Extract embeddings: {extract_embeddings}")
    print(f"    Run umap: {umap_embeddings}")
    print(f"    Refine labels in postprocessing: {refine_labels}")



    azimuth.process_query(
            normalization_override = normalization_override,
            norm_check_batch_size = norm_check_batch_size
        )

    _ = azimuth.run_inference_model()

    _ = azimuth.process_outputs(mode=output_mode)

    if refine_labels:
        _ = azimuth.refine_labels(refine_level='broad')
        medium_labels = azimuth.refine_labels(refine_level='medium')
        _ = azimuth.refine_labels(refine_level='fine')
        _ = azimuth.add_refined_score(medium_labels)

    _ = azimuth.update_cells_meta()

    if extract_embeddings:
        azimuth_embedding_layer_name = azimuth.model_meta[
                    'inference_model_embedding_layer'
            ]
        if umap_embeddings:
            (
                azimuth_embeddings, 
                azimuth_umap
                ) = azimuth.inference_embeddings_and_umap(
                    embedding_layer_name = azimuth_embedding_layer_name,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    metric=metric,
                    min_dist=min_dist,
                    umap_lr=umap_lr,
                    umap_seed=umap_seed,
                    spread=spread,
                    verbose=verbose,
                    init=init
                )

            embed_key_og = (
                f'{azimuth.inference_model_name}_'
                f'{azimuth_embedding_layer_name}_embed'
            )
            umap_key_og = (
                f'{azimuth.inference_model_name}_'
                f'{azimuth_embedding_layer_name}_umap'
            )

            azimuth.embeddings['azimuth_embed'] = azimuth_embeddings
            azimuth.umaps['azimuth_umap'] = azimuth_umap

            del azimuth.embeddings[embed_key_og]
            del azimuth.umaps[umap_key_og]

        else:
            azimuth_embeddings = azimuth.inference_model_embeddings(
                embedding_layer_name = azimuth_embedding_layer_name
            )

            azimuth.embeddings['azimuth_embed'] = azimuth_embeddings
            del azimuth.embeddings[
                f'{azimuth.inference_model_name}_'
                f'{azimuth_embedding_layer_name}_embed'
                ]

    core_outputs = {
        'azimuth_object' : azimuth,
        'embeddings_dict' : azimuth.embeddings,
        'umap_dict' : azimuth.umaps,
        'cells_meta' : azimuth.cells_meta
    }

    return core_outputs





############################ arg parsing ###############################

def arg_parse_in():
    """
    Parse command line arguments for the Azimuth cell annotation tool.
    
    Sets up argument parser with all parameters required for the annotation
    pipeline, including input file handling, model configuration, and 
    visualization options.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """

    print("Parsing arguments... \n")
    print("\n")


    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "filepath", 
                        help=(
                            "enter abs file path to the query."
                                " Query should be in h5ad format."
                        ), 
                        type=str
                        )        
    
    
    parser.add_argument(
                        "-fn",
                        "--feature_names_col",
                        default=None,
                        help=(
                            "enter the column name where the "
                                "feature names are stored in query.var"
                                " where query is the anndata object read "
                                "from the h5ad."
                        ),
                        type=str
                        )

    parser.add_argument(
                        "-ap",
                        "--annotation_pipeline", 
                        default='supervised',
                        help=(
                            "enter annotation pipeline"
                        ), 
                        type=str
                        )
    
    parser.add_argument(
                        "-ebs",
                        "--eval_batch_size", 
                        default=8192,
                        help=(
                            "enter the evaluation batch size suitable to "
                            "your hardware, defaults to 8192"
                        ), 
                        type=int
                        )
    
    parser.add_argument(
                        "-norm",
                        "--normalization_override", 
                        action = "store_true",
                        help="Override normalisation."
                        )

    parser.add_argument(
                        "-ncbs",
                        "--norm_check_batch_size", 
                        default=100,
                        help=(
                            "enter the number of cells over which "
                            "normalization will be verified, defaults "
                            "to 1000"
                        ), 
                        type=int
                        )

    parser.add_argument(
                        "-om",
                        "--output_mode", 
                        default='minimal',
                        help=(
                            "enter output mode, must be either 'minimal'"
                            " or 'detailed'."
                        ), 
                        type=str
                        )

    parser.add_argument(
                        "-rf",
                        "--refine_labels",
                        action = "store_false",
                        help="Skip label refinement."
                        )
    
    parser.add_argument(
                        "-em",
                        "--extract_embeddings",
                        action = "store_false",
                        help="Skip embedding extraction"
                        )
                        
    
    parser.add_argument(
                        "-umap",
                        "--umap_embeddings",
                        action = "store_false",
                        help="Skip UMAP creation"
                        )
    
    
    
    parser.add_argument(
                        "-nnbrs",
                        "--n_neighbors",
                        default=30,
                        help=(
                            "n_neighbors param for umaps, defaults "
                                "to Seurat default 30"
                        ),
                        type=int
                        )
    
    parser.add_argument(
                        "-nc",
                        "--n_components",
                        default=2,
                        help=(
                            "n_components param for umaps, defaults "
                            "to Seurat default 2"
                        ),
                        type=int
                        )
    
    parser.add_argument(
                        "-me",
                        "--metric",
                        default='cosine',
                        help=(
                            "metric param for umaps, defaults to "
                            "Seurat default 'cosine'"
                        ),
                        type=str
                        )
    
    parser.add_argument(
                        "-mdt",
                        "--min_dist",
                        default=0.3,
                        help=(
                            "min_dist param for umaps, defaults to "
                            "Seurat default 0.3"
                        ),
                        type=float
                        )
    
    parser.add_argument(
                        "-ulr",
                        "--umap_lr",
                        default=1.0,
                        help=("learning_rate param for umaps, defaults "
                                "to Seurat default 1.0"
                        ),
                        type=float
                        )
    
    parser.add_argument(
                        "-useed",
                        "--umap_seed",
                        default=42,
                        help=(
                            "random_state param for reproducibility of "
                            "umaps, defaults to Seurat default 42"
                        ),
                        type=int
                        )
    
    parser.add_argument(
                        "-sp",
                        "--spread",
                        default=1.0,
                        help=(
                            "spread param for umaps, defaults to "
                            "Seurat default 1.0"
                        ),
                        type=float
                        )
    
    parser.add_argument(
                        "-uv",
                        "--umap_verbose",
                        action="store_false",
                        help="Hide UMAP progress"
                        )
    
    parser.add_argument(
                        "-uin",
                        "--umap_init",
                        default="spectral",
                        help=(
                            "init param for umaps, defaults to "
                            "'spectral', the other option is 'random'"
                        ),
                        type=str
                        )
    
    parser.set_defaults(
        normalization_override=False,
        refine_labels=True,
        extract_embeddings=True,
        umap_embeddings=True,
        umap_verbose=True
    )
    
    
    args = parser.parse_args()

    return args




def arg_parse_out(args):
    """
    Convert parsed arguments to a dictionary for the annotation pipeline.
    
    Takes the parsed command line arguments and transforms them into a
    structured dictionary that can be passed to the annotation functions.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from arg_parse_in()
    
    Returns
    -------
    dict
        Dictionary of arguments ready for use in annotation functions
    """
    print("Reading arguments... \n")
    print("\n")



    query_filepath = args.filepath
    feature_names_col = args.feature_names_col
    annotation_pipeline = args.annotation_pipeline
    eval_batch_size = args.eval_batch_size
    normalization_override = args.normalization_override
    norm_check_batch_size = args.norm_check_batch_size
    output_mode = args.output_mode
    refine_labels = args.refine_labels
    extract_embeddings = args.extract_embeddings
    umap_embeddings = args.umap_embeddings
    n_neighbors = args.n_neighbors
    n_components = args.n_components
    metric = args.metric
    min_dist = args.min_dist
    umap_lr = args.umap_lr
    umap_seed = args.umap_seed
    spread = args.spread
    umap_verbose = args.umap_verbose
    umap_init = args.umap_init
    
    arguments={
        'query_filepath' : query_filepath,
        'feature_names_col' : feature_names_col,
        'annotation_pipeline' : annotation_pipeline,
        'eval_batch_size' : eval_batch_size,
        'normalization_override' : normalization_override,
        'norm_check_batch_size' : norm_check_batch_size,
        'output_mode' : output_mode,
        'refine_labels' : refine_labels,
        'extract_embeddings' : extract_embeddings,
        'umap_embeddings' : umap_embeddings,
        'n_neighbors' : n_neighbors,
        'n_components' : n_components,
        'metric' : metric,
        'min_dist' : min_dist,
        'umap_lr' : umap_lr,
        'umap_seed' : umap_seed,
        'spread' : spread,
        'verbose' : umap_verbose,
        'init' : umap_init
    }

    return arguments


############## annotate, executable python function ####################


def annotate():
    """
    Main entry point for command-line execution of the Azimuth cell 
    annotation pipeline.
    
    Parses command line arguments, loads the specified h5ad file, runs 
    the annotation pipeline via annotate_core(), and saves the results 
    as a new h5ad file in the same directory as the input file with 
    '_ANN' appended to the filename.
    
    This function is intended to be called when the module is executed 
    directly as a script and provides a complete workflow from argument 
    parsing to saving results.
    
    No parameters or return values as this function is designed to be
    the executable entry point for command-line usage.
    """
    
    args = arg_parse_in()
    arguments = arg_parse_out(args)

    for key, value in arguments.items():
        globals()[key] = value

    query_obj = ReadQueryObj(query_filepath)
    X_query = query_obj.X_query()
    query_features = query_obj.query_features(
        feature_names_col=feature_names_col
        )
    features_meta = query_obj.features_meta()
    cells_meta = query_obj.cells_meta()

    core_outputs = annotate_core(
        X_query,
        query_features,
        cells_meta,
        annotation_pipeline,
        eval_batch_size,
        normalization_override,
        norm_check_batch_size,
        output_mode,
        refine_labels,
        extract_embeddings,
        umap_embeddings,
        n_neighbors, 
        n_components, 
        metric, 
        min_dist, 
        umap_lr, 
        umap_seed, 
        spread,
        verbose,
        init
        )

    azimuth_object = core_outputs['azimuth_object']
    azimuth_object.features_meta = features_meta

    dir_path = os.path.dirname(query_filepath)
    filename = os.path.basename(query_filepath)
    filename, ext = os.path.splitext(filename)
    out_path = os.path.join(dir_path, f'{filename}_ANN{ext}')

    azimuth_object.pack_adata(save_path=out_path)


    #################################################################### 

if __name__=="__main__":
    annotate()







    












    





        















    





    

    