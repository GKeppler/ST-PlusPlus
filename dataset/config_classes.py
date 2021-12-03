import os
from ALS4L.deep_learning_components.callbacks.callback_compose import CallbackCompose
from argparse import Namespace

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ALS4L.deep_learning_components.data_processing.augmentations.semantic_segmentation.augmentation_segmentation import ImgSegProcessingPipeline, SemanticSegmentationProccesor
from ALS4L.utils.configuration.mappings import get_backbone, get_datamodule, get_labeler, get_learning_rate_scheudler, get_loss_function, get_model, get_optimizer, get_query_strategy
from ALS4L.utils.configuration.parameter_split import dict_to_config, split_parameters
import logging

def append_to_transforms(
        img_processing_args,
        train_transforms,
        val_transforms,
        test_transforms,
        val_augs=None):
    img_processing_pipeline = ImgSegProcessingPipeline(Namespace(
                                **img_processing_args))
    if val_augs is not None:
        img_processing_pipeline.make_val_aug_transform(Namespace(**val_augs))

    train_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_train_transform())
    )
    val_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_val_transform())
    )
    test_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_val_transform())
    )


class ModelBuilder():

    def __init__(self, model_params: dict, checkpoint_path: str = None):
        model_args = split_parameters(dict_to_config(model_params), ["parameters"])["parameters"]
        loss_fcn = split_parameters(dict_to_config(model_args),['loss_fcn'])
        model_args = loss_fcn['other']
        self.loss_fcn = None
        if "loss_fcn" in loss_fcn:
            loss_fcn = loss_fcn['loss_fcn']
            loss_class = get_loss_function(loss_fcn["type"])
            loss_fcn_args = split_parameters(dict_to_config(loss_fcn))
            if 'params' in loss_fcn_args:
                self.loss_fcn = loss_class(**loss_fcn_args['params'])
                model_args["loss_fcn"] = self.loss_fcn
            else:
                model_args["loss_fcn"] = loss_class()    
        backbone_config = split_parameters(dict_to_config(model_params), ["backbone"])
        if "backbone" in backbone_config:
            backbone_config = backbone_config["backbone"]
            backbone_args = split_parameters(dict_to_config(backbone_config), ["parameters"])["parameters"]
            if "checkpoint_prefix" in backbone_args:
                if backbone_args["checkpoint_prefix"] is not None and \
                    backbone_args["checkpoint_path"] is not None and \
                    backbone_args["checkpoint_path"].lower() != 'imagenet':
                    backbone_args["checkpoint_path"] = os.path.join(
                        backbone_args["checkpoint_prefix"],
                        backbone_args["checkpoint_path"]
                    )
                del backbone_args["checkpoint_prefix"]
            if "checkpoint_path" in backbone_args:
                if backbone_args["checkpoint_path"] is not None:
                    logging.info(f"Loading pre-trained model with weights: {backbone_args['checkpoint_path']}")
    
            model_args["backbone"] = get_backbone(backbone_config["name"])(**backbone_args)
        model_type = get_model(model_params["name"])
        if checkpoint_path == None:
            self.instance = model_type(**(model_args))
        else:
            self.instance = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path, **(model_args))

        optimizer_config = split_parameters(dict_to_config(model_params), ["optimizer"])
        if "optimizer" in optimizer_config:
            optimizer_config = optimizer_config["optimizer"]
            optimizer = get_optimizer(optimizer_config['type'])
            optimizer_params_config = split_parameters(dict_to_config(optimizer_config))
            #test 
            grouped_parameters = []
            logging.info('Initializing Optimizers.')
            logging.info(f'Default Params: {str(optimizer_params_config["params"])}')
            for cm in self.instance.named_children():
                child_module_name = cm[0]
                child_module = cm[1]
                # if the child moudle name is in config it contains specific configigurations
                if child_module_name in optimizer_params_config.keys():
                    specific_config = split_parameters(optimizer_params_config[child_module_name])
                    logging.info(f'Custom config for {child_module_name}. Params: {str(specific_config["params"])}')
                    params = {
                        "params": [p[1] for p in child_module.named_parameters()],
                         **specific_config['params']
                    }
                else:
                    params = {
                        "params": [p[1] for p in child_module.named_parameters()],
                         **optimizer_params_config['params']
                    }
                if len(params['params']) > 0:
                    grouped_parameters.append(params)
            self.optim_instance = optimizer(
                params=grouped_parameters,
                **optimizer_params_config['params']
            )
            lrs = None
            self.lrs_instance = None
            if "lrs" in optimizer_params_config:
                lrs_config = optimizer_params_config['lrs']
                lrs = get_learning_rate_scheudler(lrs_config['type'])
                lrs_params_config = split_parameters(dict_to_config(lrs_config))
                self.lrs_instance = lrs(
                    self.optim_instance,
                    **lrs_params_config['params']
                )
            if self.lrs_instance is not None:
                self.instance.set_optimizers(
                    optimizer=self.optim_instance,
                    lrs=self.lrs_instance
                )
            else:
                self.instance.set_optimizers(
                    optimizer=self.optim_instance
                )            


class DataModuleBuilder():

    def __init__(self, data_params: dict,  do_val_init=True):
        datamodule_params = split_parameters(
                                    dict_to_config(data_params),
                                    ["datamodule"])["datamodule"]
        datamodule_args = split_parameters(
                                    dict_to_config(datamodule_params),
                                    ["arguments"])["arguments"]
        img_processing_args = split_parameters(
                                    dict_to_config(data_params),
                                    ["img_processing"])["img_processing"]

        transformations = split_parameters(dict_to_config(img_processing_args))
        general_configuration = transformations['other']
        del transformations['other']

        # extract additional validation time transforms
        val_augs = None
        if 'validation_aug' in transformations:
            val_augs = transformations['validation_aug']
            del transformations['validation_aug']

        train_transforms = []
        val_transforms = []
        test_transforms = []
        if len(transformations.values()) > 0:
            for img_processing in transformations.values():
                append_to_transforms(
                                    dict(general_configuration, **img_processing),
                                    train_transforms,
                                    val_transforms,
                                    test_transforms,
                                    val_augs)
        else:
            append_to_transforms(
                                general_configuration,
                                train_transforms,
                                val_transforms,
                                test_transforms,
                                val_augs)

        datamodule = get_datamodule(datamodule_params["name"])
        # Create new dict with train_transforms and val_transforms
        # and merge it with the datamodule args.
        datamodule = datamodule(
            **{
                **{
                    **{
                        "train_transforms": train_transforms,
                        "val_transforms": val_transforms,
                        "test_transforms": test_transforms,
                        "root_dir":
                            datamodule_params["root_dirs"]
                            [datamodule_params["device"]],
                    },
                    **datamodule_args,
                }
            }
        )
        active_learing_params = split_parameters(
                                    dict_to_config(data_params),
                                    ["active_learning"])["active_learning"]

        #datamodule.assign_labeled_unlabeled_split()                            

        if active_learing_params:
            labeler = get_labeler(active_learing_params['labeler'])()
            ranker = get_query_strategy(active_learing_params['query_strategy'])()
            samples_to_label = active_learing_params['samples_to_label']
            labeler.label_samples(
                datamodule, 
                samples_to_label, 
                ranker.rank_samples_to_label(datamodule.train_dataloader(get_labeled_share=False).dataset)
            )

        if hasattr(datamodule, 'init_val_dataset') and do_val_init:
            datamodule.init_val_dataset()

        self.instance = datamodule

        logging.info(f"Length of labeled train dataset: {len(self.instance.labeled_train_dataset)}")
        logging.info(f"Length of unlabeled train dataset: {len(self.instance.unlabeled_train_dataset)}")
        if do_val_init:
            logging.info(f"Length of validation dataset: {len(self.instance.val_dataset)}")
            logging.info(f"Length of test dataset: {len(self.instance.test_dataset)}")


class TrainerBuilder():

    def __init__(self, train_params: dict, result_dir: str, run_name: str, data:DataModuleBuilder = None):
        callback_args = split_parameters(
            dict_to_config(train_params),
            ["callbacks"]
        )["callbacks"]
        callback_args['experiment_dir'] = result_dir
        trainer_args = split_parameters(
            dict_to_config(train_params),["trainer"])["trainer"]
       
        logger = TensorBoardLogger(
            save_dir=result_dir,
            name=run_name,
        )
        
        self.callbacks = CallbackCompose(Namespace(**callback_args),data)
        self.instance = Trainer(
            **trainer_args,
            default_root_dir=result_dir,
            logger=logger,
            callbacks=self.callbacks.get_composition()
        )