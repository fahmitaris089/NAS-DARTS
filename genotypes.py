"""
Genotype — Architecture Encoding
==================================
A Genotype encodes the discovered cell topology:
  - Which operations are selected per edge
  - Which input nodes each intermediate node reads from
  - Which intermediate nodes are concatenated for cell output
"""

from collections import namedtuple

Genotype = namedtuple('Genotype', [
    'normal',         # list of (op_name, input_node_idx)
    'normal_concat',  # list of node indices to concat
    'reduce',         # list of (op_name, input_node_idx)
    'reduce_concat',  # list of node indices to concat
])


def genotype_to_dict(genotype):
    """Convert Genotype namedtuple to JSON-serialisable dict."""
    return {
        'normal':        list(genotype.normal),
        'normal_concat': list(genotype.normal_concat),
        'reduce':        list(genotype.reduce),
        'reduce_concat': list(genotype.reduce_concat),
    }


def dict_to_genotype(d):
    """Reconstruct Genotype from dict (e.g. loaded from JSON)."""
    return Genotype(
        normal=        [tuple(e) for e in d['normal']],
        normal_concat= list(d['normal_concat']),
        reduce=        [tuple(e) for e in d['reduce']],
        reduce_concat= list(d['reduce_concat']),
    )


# ─── Reference Genotypes (for comparison / sanity check) ─────────────────────

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('skip_connect', 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 2), ('max_pool_3x3', 1),
        ('max_pool_3x3', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('max_pool_3x3', 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('skip_connect', 0), ('dil_conv_3x3', 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0), ('max_pool_3x3', 1),
        ('skip_connect', 2), ('max_pool_3x3', 1),
        ('max_pool_3x3', 0), ('skip_connect', 2),
        ('skip_connect', 2), ('max_pool_3x3', 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)
