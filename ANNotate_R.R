#!/usr/bin/env Rscript 


#### function to check availability of an nvidia gpu through nvidia-smi
if_gpu <- function() {
  gpu_check <- system("which nvidia-smi", intern = TRUE, ignore.stderr = TRUE)
  
  if (length(gpu_check) == 0) {
    return(FALSE)
  } else{
    return(TRUE)
  }
}


#### creating and setting up conda environment
if (!requireNamespace("reticulate", quietly = TRUE) || 
    packageVersion("reticulate") < "1.40.0") {
  message("Installing or updating reticulate to version 1.40.0 alongwith dependencies Rcpp and RcppTOML...")
  install.packages("Rcpp")
  install.packages("RcppTOML")
  install.packages("reticulate")
  
  if (!requireNamespace("reticulate", quietly = TRUE) || 
      packageVersion("reticulate") < "1.40.0") {
    stop("Failed to install or update reticulate to version 1.40.0. Please resolve this issue externally.")
  } else {
    message("reticulate successfully updated to version 1.40.0.")
  }
} else {
  message("reticulate is already installed and up to date.")
}
library(reticulate)

#### create conda env and install dependencies
setup_conda_env <- function(yml_file, requirements_file, conda_path=NULL, force_create=FALSE) {
  
  if (is.null(conda_path)) {
    conda_path <- reticulate::conda_binary()
  }
  if (is.null(conda_path)) {
    stop("Conda not found. Please install conda or miniconda and try again.")
  }
  
  env_name <- "AzimuthNN_test" 
  
  env_check_command <- sprintf('%s env list | grep "%s"', conda_path, env_name)
  env_check <- system(env_check_command, intern = TRUE)
  
  if (force_create && length(env_check) > 0) {
    message(sprintf("Conda environment '%s' exists, but force_create is TRUE. Deleting it first...", env_name))
    system2(conda_path, c("env", "remove", "--name", env_name, "--yes"))
    message(sprintf("Environment '%s' deleted successfully.", env_name))
    env_check <- character(0) 
  }
  
  
  if (length(env_check) == 0) {
    message(sprintf("Conda environment '%s' does not exist. Creating it first...", env_name))
    
    if (!file.exists(yml_file)) {
      stop(sprintf("YML file does not exist at the path: %s", yml_file))
    }
    
    library(yaml)
    yml_data <- yaml::read_yaml(yml_file)
    env_name <- yml_data$name
    
    message(sprintf("Creating conda environment '%s' from '%s'...", env_name, yml_file))
    
    cmd <- c("env", "create", "-f", yml_file) 
    message(sprintf("Running command: %s", paste(cmd, collapse = " ")))
    
    
    system2(conda_path, cmd)
    
    message(sprintf("Environment '%s' created successfully.", env_name))
    
    reticulate::use_condaenv(condaenv = env_name, conda = conda_path, required = TRUE)
    message(sprintf("Environment '%s' is now active.", env_name))
    
    active_env <- reticulate::py_config()$python
    expected_env <- file.path(dirname(dirname(conda_path)), "envs", env_name, "bin", "python")
    
    if (normalizePath(active_env) != normalizePath(expected_env)) {
      stop(sprintf(
        "Error: Reticulate is not connected to the expected conda environment properly.\nExpected: %s\nActive: %s",
        expected_env, active_env
      ))
    } else {
      message(sprintf("Reticulate is correctly connected to: %s", active_env))
    }
    
    if (!is.null(requirements_file) && file.exists(requirements_file)) {
      message(sprintf("Installing pip dependencies from %s...", requirements_file))
      system2(conda_path, c("run", "-n", env_name, "pip", "install", "-r", requirements_file))
      system2(conda_path, c("run", "-n", env_name, "pip", "install", "--upgrade", "pip"))
      system2(conda_path, c("run", "-n", env_name, "pip", "install", "--upgrade", "keras"))
      if (if_gpu()){
        system2(conda_path, c("run", "-n", env_name, "pip", "install", "tensorflow[and-cuda]==2.17"))
      }else{
        system2(conda_path, c("run", "-n", env_name, "pip", "install", "tensorflow-cpu==2.17.0"))
      }
      
      message("All pip dependencies installed successfully.")
    } else {
      message("No requirements file provided or found.")
    }
    
    print(reticulate::py_config())
  } else {
    message(sprintf("Conda environment '%s' found.", env_name))
    
    reticulate::use_condaenv(condaenv = env_name, conda = conda_path, required = TRUE)
    message(sprintf("Environment '%s' is now active.", env_name))
    
    active_env <- reticulate::py_config()$python
    expected_env <- file.path(dirname(dirname(conda_path)), "envs", env_name, "bin", "python")
    
    if (normalizePath(active_env) != normalizePath(expected_env)) {
      stop(sprintf(
        "Error: Reticulate is not connected to the expected conda environment properly.\nExpected: %s\nActive: %s",
        expected_env, active_env
      ))
    } else {
      message(sprintf("Reticulate is correctly connected to: %s", active_env))
    }
    
    print(reticulate::py_config())
  }
  
}

yml_path <- "/brahms/sarkars/Panhuman_AzimuthNN_public/utils/env_min.yml" 
requirements_path <- "/brahms/sarkars/Panhuman_AzimuthNN_public/utils/requirements_min.txt" 

#conda_binary <- "/path/to/conda/bin/conda" 
setup_conda_env(yml_file = yml_path, requirements_file = requirements_path)


#### R dependencies
library(Seurat)
library(Matrix)

ensure_argparse <- function() {
  if (!requireNamespace("argparse", quietly = TRUE)) {
    message("The 'argparse' package is not installed. Installing now...")
    install.packages("argparse")
    message("'argparse' package installed successfully.")
  } else {
    message("'argparse' package is already installed.")
  }
}

ensure_argparse()
library(argparse)

#### other python dependencies
python_module_path <- "/brahms/sarkars/Panhuman_AzimuthNN_public"
py_run_string(paste("import sys; sys.path.append('", python_module_path, "')", sep = ""))
annotate <- import("ANNotate")
sp <- import("scipy.sparse")

#### functions

get_data <- function(object, assay, layer= "data") {
  if (packageVersion("Seurat") >= "5.0.0") {
    return(LayerData(object[[assay]], layer = layer))
  } else {
    assay_obj <- object@assays[[assay]]
    return(slot(assay_obj, layer))
  }
}


read_obj_min <- function(query_obj, feature_names_col, assay_default='RNA') {
  
  if (!(assay_default %in% names(query_obj@assays))){
    stop(paste("Assay", assay_default, "not present in the seurat object."))
  }
  
  DefaultAssay(query_obj) <- assay_default
  
  if (packageVersion("Seurat") >= "5.0.0"){
    layers_obj <- Layers(query_obj, assay = assay_default)
  } else{
    layers_obj <- slotNames(query_obj[[assay_default]])
  }
  
  if ("data" %in% layers_obj) {
    normalized_data <- get_data(query_obj, assay = assay_default)
  } else {
    query_obj <- NormalizeData(query_obj[[assay_default]])
    normalized_data <- get_data(query_obj, assay = assay_default)
  }
  
  X_query <- t(normalized_data)
  X_query <- Matrix(X_query, sparse = TRUE)
  
  cell_metadata <- query_obj@meta.data
  query_cells_df <- as.data.frame(cell_metadata)
  
  if (!is.null(feature_names_col)) {
    feature_metacols <- colnames(query_obj[[assay_default]][[]])
    if (feature_names_col %in% feature_metacols){
      query_features <- query_obj[[assay_default]][[feature_names_col]]
      query_features <- as.list(query_features[[feature_names_col]])
    } else {
      stop(paste(feature_names_col, "not found as a column in the df returned by object[[",assay_default,"]][[]]"))
    }
    
  } else {
    query_features <- as.list(rownames(normalized_data))
  }
  return(list(X_query = X_query, query_features = query_features, query_cells_df = query_cells_df))
}

package_obj <- function(embeddings_mode, embeddings_dict, if_umap_embeddings, umap_embeddings_dict, query_cells_df, query_obj) {
  
  if (!is.null(embeddings_mode)) {
    for (em_name in names(embeddings_dict)) {
      
      em_matrix <- as.matrix(embeddings_dict[[em_name]])
      
      if (nrow(em_matrix) != length(Cells(query_obj))) {
        stop(paste("Dimension mismatch:", em_name, " does not have as many cells as the query obj."))
      }
      
      rownames(em_matrix) <- Cells(query_obj)
      
      dimreduc_obj <- CreateDimReducObject(embeddings = em_matrix, key = paste0("ANN", em_name, "_"), assay = DefaultAssay(query_obj))
      
      query_obj[[paste0("ANN", em_name)]] <- dimreduc_obj
    }
  }
  
  if (if_umap_embeddings) {
    for (em_name in names(umap_embeddings_dict)) {
      
      em_matrix <- as.matrix(umap_embeddings_dict[[em_name]])
      
      if (nrow(em_matrix) != length(Cells(query_obj))) {
        stop(paste("Dimension mismatch:", em_name, " does not have as many cells as the query obj."))
      }
      
      rownames(em_matrix) <- Cells(query_obj)
      
      dimreduc_obj <- CreateDimReducObject(embeddings = em_matrix, key = paste0("umapANN", em_name, "_"), assay = DefaultAssay(query_obj))
      
      query_obj[[paste0("umapANN", em_name)]] <- dimreduc_obj
    }
  }
  
  query_obj@meta.data <- as.data.frame(query_cells_df)  
  
  return(query_obj)
}

PrepLabel <- function(object, label_id = 'final_level_label', newid = 'PrepLabel', cutid = 'Other', cutoff=10) {
  rejected_names <- names(which(table(object@meta.data[,label_id])<cutoff))
  object@meta.data[,newid]=as.character(object@meta.data[,label_id])
  rejected_cells <- which(object@meta.data[,label_id]%in%rejected_names)
  object@meta.data[rejected_cells,newid]=cutid
  return(object)
}




ANNotate <- function(
                    query_obj,
                    feature_names_col=NULL,
                    source_data_dir="/brahms/sarkars/AzimuthNN_clone/AzimuthNN/sarkars/data/dataset_main",
                    features_txt="features_02_26_25_17_50.txt",
                    split_mode="cumulative",
                    model="M0.2",
                    loss_name="level_wt_focal_loss",
                    epochs=55,
                    train_seed=100,
                    data_seed=414,
                    data_source="data/kfold_data/datasets/fold10_02_26_2025_17_53_139",
                    data_split=c(7,1,2),
                    mask_seed=NULL,
                    tm_frac=NULL,
                    lm_frac=NULL,
                    save=TRUE,
                    batch_size=256,
                    eval_batch_size=40960,
                    optimizer_name="adam",
                    lr=NULL,
                    l1=NULL,
                    l2=0.01,
                    dropout=0.1,
                    normalization_override=FALSE,
                    embeddings_mode="shallow",
                    if_knn_scores=FALSE,
                    if_umap_embeddings=TRUE,
                    if_refine_labels=TRUE,
                    n_neighbors=30,
                    n_components=2,
                    metric="cosine",
                    min_dist=0.3,
                    umap_lr=1.0,
                    umap_seed=42,
                    spread=1.0,
                    verbose=TRUE,
                    init="spectral",
                    object_disk=FALSE,
                    out_file_disk=FALSE,
                    process_obj=TRUE,
                    cutoff_abs=5,
                    cutoff_frac=0.001){

  options(warn = -1)
  
  cat("Running Pan-Human Azimuth:\n")
  cat("\n")
  
  #### make sure that integers are passed to python as integers
  epochs <- as.integer(epochs)
  train_seed <- as.integer(train_seed)
  data_seed <- as.integer(data_seed)
  data_split <- lapply(data_split, as.integer)
  if (!is.null(mask_seed)){
    mask_seed <- as.integer(mask_seed)
  }
  batch_size <- as.integer(batch_size)
  eval_batch_size <- as.integer(eval_batch_size)
  n_neighbors <- as.integer(n_neighbors)
  n_components <- as.integer(n_components)
  umap_seed <- as.integer(umap_seed)

   
  
  #### reading the seurat object 
  query <- read_obj_min(query_obj, feature_names_col)
  X_query <- sp$csr_matrix(r_to_py(query$X_query))
  query_features <- query$query_features
  query_cells_df <- query$query_cells_df

  
  
  
  #### run annotate_core
  core_outputs <- annotate$annotate_core(
    X_query,
    query_features,
    source_data_dir,
    features_txt,
    split_mode,
    model,
    epochs,
    train_seed,
    loss_name,
    data_seed,
    data_source,
    data_split,
    mask_seed,
    tm_frac,
    lm_frac,
    batch_size,
    optimizer_name,
    lr,
    l1,
    l2, 
    dropout,
    save,
    eval_batch_size,
    normalization_override,
    embeddings_mode,
    query_cells_df,
    if_knn_scores,
    if_umap_embeddings,
    if_refine_labels,
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
  
  
  embeddings_mode <- core_outputs[[3]]
  embeddings_dict <- core_outputs[[4]]
  query_cells_df <- core_outputs[[10]]
  if_umap_embeddings <- core_outputs[[11]]
  umap_embeddings_dict <- core_outputs[[12]]
  
  annotated_obj = package_obj(embeddings_mode, embeddings_dict, if_umap_embeddings, umap_embeddings_dict, query_cells_df, query_obj)
  
  
  
  if (process_obj){
    annotated_obj <- PrepLabel(annotated_obj,label_id = 'final_level_label',cutoff = min(cutoff_abs, cutoff_frac*ncol(annotated_obj)),cutid = 'Other',newid = 'azimuth_label')
    Idents(annotated_obj) <- 'azimuth_label'
  }
  
  
  
  return(annotated_obj)
  
}





args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  cat("\n")
  cat("ANNotate_R is executed from the terminal...")
  annotate_R() 
} else {
  cat("\n")
  cat("ANNotate_R is being run interactively.")
}
