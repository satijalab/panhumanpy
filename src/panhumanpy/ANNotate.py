"""
Inference script based on the Azimuth Neural Network trained
on annotated panhuman scRNA-seq data.
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
    object for low-level interactive usage
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
        Load query data directly.
        
        Args:
            X_query: scipy.sparse.csr_matrix, Expression matrix
            query_features: list of str, Feature names
            cells_meta: pandas.DataFrame, Cell metadata
            
        Raises:
            TypeError: If inputs are not of correct type
            ValueError: If dimensions don't match
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
        reading a query AnnData obj
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
        reading a query from disk
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
        norm_check_batch_size=1000
        ):

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
            self.processed_outputs['all_level_labels'] = (
                output_processing_class.all_level_labels()
            )

        return self.processed_outputs

    def refine_labels(self, refine_level):

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

    def inference_model_embeddings(self, embedding_layer_name):

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
    object for high level interactive usage 
    """

    def __init__(
        self,
        query_arg,
        feature_names_col=None,
        annotation_pipeline='supervised',
        eval_batch_size=8192,
        normalization_override=False,
        norm_check_batch_size=1000,
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

        _ = self.refine_labels(refine_level='broad')
        _ = self.refine_labels(refine_level='medium')
        _ = self.refine_labels(refine_level='fine')

        _ = self.update_cells_meta()

    def azimuth_embed(self):

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
    ):

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
        _ = azimuth.refine_labels(refine_level='medium')
        _ = azimuth.refine_labels(refine_level='fine')

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
        Arg parser to collect arguments to pass to ANNotate.py
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
                            default=False,
                            help=(
                                "is the counts data lop1p normalized after "
                                "scaling to 10k? defaults to False"
                            ), 
                            type=bool
                            )

        parser.add_argument(
                            "-ncbs",
                            "--norm_check_batch_size", 
                            default=1000,
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
                            default=True,
                            help=(
                                "do you want to refine annotations? "
                                "default is True"
                            ),
                            type=bool,
                            )
        
        parser.add_argument(
                            "-em",
                            "--extract_embeddings",
                            default=True,
                            help=(
                                "extract embeddings? defaults to True"
                            ),
                            type=bool
                            )
                            
        
        parser.add_argument(
                            "-umap",
                            "--umap_embeddings",
                            default=True,
                            help=(
                                "specify if you want umap embeddings, "
                                "defaults to False"
                            ),
                            type=bool
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
                            default=True,
                            help=("verbose param for umaps, "
                                  "defaults to True"
                            ),
                            type=bool
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
        
        
        args = parser.parse_args()

        return args




def arg_parse_out(args):
        """
        Reading the arguments passed to the arg parser for ANNotate.py.
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







    












    





        















    





    

    