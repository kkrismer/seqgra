"""
MIT - CSAIL - Gifford Lab - seqgra

Implementation of Parser for XML configuration files 
(using Strategy design pattern)

@author: Konstantin Krismer
"""
import io
import logging
from typing import Dict, List, Any
from xml.dom.minidom import Document, parseString

import pkg_resources
from lxml import etree

from seqgra.parser import XMLHelper
from seqgra.parser import ModelDefinitionParser
from seqgra.model import ModelDefinition
from seqgra.model.model import Architecture
from seqgra.model.model import Operation


class XMLModelDefinitionParser(ModelDefinitionParser):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def __init__(self, config: str) -> None:
        self._dom: Document = parseString(config)
        self._general_element: Any = \
            self._dom.getElementsByTagName("general")[0]
        XMLModelDefinitionParser.validate(config)

    @staticmethod
    def validate(xml_config: str) -> None:
        xsd_path = pkg_resources.resource_filename("seqgra",
                                                   "model-config.xsd")
        xmlschema_doc = etree.parse(xsd_path)
        xmlschema = etree.XMLSchema(xmlschema_doc)
        xml_doc = etree.parse(io.BytesIO(xml_config.encode()))
        xmlschema.assertValid(xml_doc)
        logging.info("seqgra model configuration XML file "
                     "is well-formed and valid")

    def get_model_id(self) -> str:
        return self._general_element.getAttribute("id")

    def get_name(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "name")

    def get_description(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "description")
        
    def get_task(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "task")
        
    def get_sequence_space(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "sequencespace")

    def get_library(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "library")

    def get_implementation(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "implementation")

    def get_labels(self) -> List[str]:
        labels_element: Any = \
            self._general_element.getElementsByTagName("labels")[0]
        label_elements = labels_element.getElementsByTagName("label")
        return [XMLHelper.read_immediate_text_node(label_element)
                for label_element in label_elements]

    def get_seed(self) -> str:
        return XMLHelper.read_int_node(self._general_element, "seed")

    def get_architecture(self) -> Architecture:
        sequential_element = self._dom.getElementsByTagName("sequential")
        if len(sequential_element) == 1:
            operation_elements: Any = \
                sequential_element[0].getElementsByTagName("operation")
            operations = [self.__parse_operation(operation_element)
                          for operation_element in operation_elements]
        else:
            operations = None

        hyperparameters_element = \
            self._dom.getElementsByTagName("hyperparameters")
        if len(hyperparameters_element) == 1:
            hyperparameter_elements: Any = \
                hyperparameters_element[0].getElementsByTagName(
                    "hyperparameter")
            hyperparameters = \
                self.__parse_hyperparameters(hyperparameter_elements)
        else:
            hyperparameters = None

        external_element = self._dom.getElementsByTagName("external")
        if len(external_element) == 1:
            external_model_path: str = \
                XMLHelper.read_immediate_text_node(external_element[0])
            external_model_format: str = \
                external_element[0].getAttribute("format")
            if external_element[0].hasAttribute("classname"):
                external_model_class_name: str = \
                    external_element[0].getAttribute("classname")
            else:
                external_model_class_name: str = None
        else:
            external_model_path: str = None
            external_model_format: str = None
            external_model_class_name: str = None

        return Architecture(operations, hyperparameters,
                            external_model_path, external_model_format,
                            external_model_class_name)

    def __parse_operation(self, operation_element) -> Operation:
        return Operation(operation_element.firstChild.nodeValue,
                         dict(operation_element.attributes.items()))

    def get_loss_hyperparameters(self) -> Dict[str, str]:
        loss_element: Any = self._dom.getElementsByTagName("loss")
        if len(loss_element) == 1:
            hyperparameter_elements: Any = \
                loss_element[0].getElementsByTagName("hyperparameter")
            return self.__parse_hyperparameters(hyperparameter_elements)
        else:
            return None

    def get_optimizer_hyperparameters(self) -> Dict[str, str]:
        optimizer_element: Any = self._dom.getElementsByTagName("optimizer")
        if len(optimizer_element) == 1:
            hyperparameter_elements: Any = \
                optimizer_element[0].getElementsByTagName("hyperparameter")
            return self.__parse_hyperparameters(hyperparameter_elements)
        else:
            return None

    def get_training_process_hyperparameters(self) -> Dict[str, str]:
        training_process_element: Any = \
            self._dom.getElementsByTagName("trainingprocess")[0]
        hyperparameter_elements: Any = \
            training_process_element.getElementsByTagName("hyperparameter")
        return self.__parse_hyperparameters(hyperparameter_elements)

    def __parse_hyperparameters(self,
                                hyperparameter_elements) -> Dict[str, str]:
        hyperparams: Dict[str, str] = dict()

        for hyperparameter_element in hyperparameter_elements:
            hyperparams[hyperparameter_element.getAttribute("name")] = \
                hyperparameter_element.firstChild.nodeValue

        return hyperparams

    def get_model_definition(self) -> ModelDefinition:
        return ModelDefinition(self.get_model_id(),
                               self.get_name(),
                               self.get_description(),
                               self.get_task(),
                               self.get_sequence_space(),
                               self.get_library(),
                               self.get_implementation(),
                               self.get_labels(),
                               self.get_seed(),
                               self.get_architecture(),
                               self.get_loss_hyperparameters(),
                               self.get_optimizer_hyperparameters(),
                               self.get_training_process_hyperparameters())
