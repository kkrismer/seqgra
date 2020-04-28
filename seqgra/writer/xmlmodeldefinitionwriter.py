"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from typing import Dict, List, Tuple, Optional

from lxml import etree

from seqgra.model import ModelDefinition
from seqgra.writer import ModelDefinitionWriter
from seqgra.model.model import Architecture
from seqgra.model.model import Operation


class XMLModelDefinitionWriter(ModelDefinitionWriter):
    @staticmethod
    def create_root_element():
        attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance",
                                 "noNamespaceSchemaLocation")
        ns_map = {"xsi": "http://www.w3.org/2001/XMLSchema-instance"}

        return etree.Element("seqgramodel",
                             {attr_qname: "https://seqgra.mit.edu/model-config.xsd"},
                             nsmap=ns_map)

    @staticmethod
    def attach_labels_element(general_element, labels: List[str]) -> None:
        labels_element = etree.SubElement(general_element, "labels")
        for label in labels:
            label_element = etree.SubElement(labels_element, "label")
            label_element.text = label

    @staticmethod
    def attach_general_element(root, model_definition: ModelDefinition) -> None:
        general_element = etree.SubElement(root, "general",
                                           {"id": model_definition.model_id})

        name_element = etree.SubElement(general_element, "name")
        name_element.text = model_definition.name

        description_element = etree.SubElement(general_element, "description")
        description_element.text = model_definition.description

        task_element = etree.SubElement(general_element, "task")
        task_element.text = model_definition.task

        sequence_space_element = etree.SubElement(general_element,
                                                  "sequencespace")
        sequence_space_element.text = model_definition.sequence_space

        library_element = etree.SubElement(general_element, "library")
        library_element.text = model_definition.library

        if model_definition.implementation is not None:
            implementation_element = etree.SubElement(general_element,
                                                    "implementation")
            implementation_element.text = model_definition.implementation

        if model_definition.labels:
            XMLModelDefinitionWriter.attach_labels_element(
                general_element, model_definition.labels)
        else:
            raise Exception("no labels specified")

        seed_element = etree.SubElement(general_element, "seed")
        seed_element.text = str(model_definition.seed)

    @staticmethod
    def attach_hp_element(hps_element, hps: Dict[str, str]) -> None:
        for hp_name, hp_value in hps.items():
            hp_element = etree.SubElement(
                hps_element, "hyperparameter", {"name": hp_name})
            hp_element.text = hp_value

    @staticmethod
    def attach_architecture_element(root, architecture: Architecture) -> None:
        if (architecture.operations or
            architecture.hyperparameters) and \
            (architecture.external_model_path or
                architecture.external_model_format or
             architecture.external_model_class_name):
            raise Exception("cannot both specify operations / "
                            "hyperparameters and external model")

        architecture_element = etree.SubElement(root, "architecture")

        if architecture.operations:
            sequential_element = etree.SubElement(
                architecture_element, "sequential")
            for operation in architecture.operations:
                if operation.parameters:
                    op_element = etree.SubElement(sequential_element,
                                                  "operation",
                                                  operation.parameters)
                else:
                    op_element = etree.SubElement(sequential_element,
                                                  "operation")
                op_element.text = operation.name

        if architecture.hyperparameters:
            hps_element = etree.SubElement(
                architecture_element, "hyperparameters")
            XMLModelDefinitionWriter.attach_hp_element(
                hps_element, architecture.hyperparameters)

        if architecture.external_model_path:
            external_element = etree.SubElement(
                architecture_element, "external",
                {"format": architecture.external_model_format})
            external_element.text = architecture.external_model_path

            if architecture.external_model_class_name:
                external_element.set("classname",
                                     architecture.external_model_class_name)

    @staticmethod
    def attach_loss_element(root, loss_hp: Optional[Dict[str, str]]) -> None:
        if loss_hp:
            loss_element = etree.SubElement(root, "loss")
            XMLModelDefinitionWriter.attach_hp_element(
                loss_element, loss_hp)

    @staticmethod
    def attach_optimizer_element(root,
                                 optimizer_hp: Optional[Dict[str, str]]) -> None:
        if optimizer_hp:
            optimizer_element = etree.SubElement(root, "optimizer")
            XMLModelDefinitionWriter.attach_hp_element(
                optimizer_element, optimizer_hp)

    @staticmethod
    def attach_tp_element(root, tp_hp: Optional[Dict[str, str]]) -> None:
        if tp_hp:
            tp_element = etree.SubElement(root, "trainingprocess")
            XMLModelDefinitionWriter.attach_hp_element(tp_element, tp_hp)

    @staticmethod
    def write_model_definition_to_file(model_definition: ModelDefinition,
                                       file_name: str):
        root = XMLModelDefinitionWriter.create_root_element()
        XMLModelDefinitionWriter.attach_general_element(root, model_definition)
        XMLModelDefinitionWriter.attach_architecture_element(
            root, model_definition.architecture)
        XMLModelDefinitionWriter.attach_loss_element(
            root, model_definition.loss_hyperparameters)
        XMLModelDefinitionWriter.attach_optimizer_element(
            root, model_definition.optimizer_hyperparameters)
        XMLModelDefinitionWriter.attach_tp_element(
            root, model_definition.training_process_hyperparameters)

        with open(file_name, "wb") as config_file:
            config_file.write(etree.tostring(root, pretty_print=True,
                                             encoding="UTF-8", method="xml",
                                             xml_declaration=True))
