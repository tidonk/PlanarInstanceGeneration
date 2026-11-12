# Queue labels are independent of result-directory names.
batch_group_ogdf_20261106_sources = ogdf_20261106
batch_variant_ogdf_20261106_sources = -sources
batch_group_ogdf_20261104_normalized = ogdf_20261104
batch_variant_ogdf_20261104_normalized = -normalized
batch_group_ogdf_20261104_piece_comparison = ogdf_20261104
batch_variant_ogdf_20261104_piece_comparison = -constituents
batch_group_ogdf_sampling_20261106_equidistant = ogdf_20261106
batch_variant_ogdf_sampling_20261106_equidistant = -equidistant
batch_group_ogdf_sampling_20261106_adaptive_skip_one = ogdf_20261106
batch_variant_ogdf_sampling_20261106_adaptive_skip_one = -adaptive-skip-one
batch_group_ogdf_sampling_20261106_prefer_non_leaf = ogdf_20261106
batch_variant_ogdf_sampling_20261106_prefer_non_leaf = -prefer-non-leaf
batch_group = $(batch_group_$(experiment):$(experiment))
batch_variant = $(batch_variant_$(experiment):)
