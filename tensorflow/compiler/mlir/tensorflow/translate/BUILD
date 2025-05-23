load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "import_model",
    srcs = [
        "import_model.cc",
    ],
    hdrs = [
        "import_model.h",
    ],
    deps = [
        ":mlir_roundtrip_flags",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:loader_util",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tensorflow/transforms:initialize_variables_in_session_init",
        "//tensorflow/compiler/mlir/tensorflow/transforms:lift_variables_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:mark_initialized_variables_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "mlir_roundtrip_flags",
    hdrs = ["mlir_roundtrip_flags.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tf2xla/api/v2:mlir_roundtrip_flags",
    ],
)

cc_library(
    name = "mlir_import_options",
    hdrs = ["mlir_import_options.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_lib",
    srcs = ["tf_mlir_translate.cc"],
    hdrs = ["tf_mlir_translate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":import_model",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)
