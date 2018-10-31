# coding: utf-8

from hmtl.models.coref_custom import CoreferenceCustom
from hmtl.models.relation_extraction import RelationExtractor

#Single Module
from hmtl.models.layerNer import LayerNer
from hmtl.models.layerRelation import LayerRelation
from hmtl.models.layerCoref import LayerCoref

#Two modules
from hmtl.models.layerNerEmd import LayerNerEmd
from hmtl.models.layerEmdRelation import LayerEmdRelation
from hmtl.models.layerEmdCoref import LayerEmdCoref

#Three modules
from hmtl.models.layerNerEmdCoref import LayerNerEmdCoref
from hmtl.models.layerNerEmdRelation import LayerNerEmdRelation

#Four modules
from hmtl.models.hmtl import HMTL