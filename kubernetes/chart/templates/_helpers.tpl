{{- define "pipeline-training.name" -}}
{{- printf "train-%s" .Release.Name | trunc 56 | trimSuffix "-" -}}
{{- end -}}

{{- define "pipeline-training.config" -}}
{{- $nodes := int .Values.numNodes -}}
{{- $gpus := int .Values.gpusPerNode -}}
{{- $dp := int .Values.runConfig.train.data_parallel_num -}}
{{- $pp := int .Values.runConfig.train.pipeline_parallel_stages -}}
{{- $micro := int .Values.runConfig.train.microbatch_num -}}
{{- if or (lt $nodes 1) (lt $gpus 1) (lt $dp 1) (lt $pp 1) (lt $micro 1) -}}
{{- fail "node, GPU, DP, PP and microbatch counts must be positive" -}}
{{- end -}}
{{- if ne (mul $dp $pp) (mul $nodes $gpus) -}}
{{- fail "data_parallel_num * pipeline_parallel_stages must equal numNodes * gpusPerNode" -}}
{{- end -}}
{{- if or (lt (int .Values.runConfig.train.batch_size) 1) (ne (mod (int .Values.runConfig.train.batch_size) (mul $dp $micro)) 0) -}}
{{- fail "batch_size must be positive and divisible by data_parallel_num * microbatch_num" -}}
{{- end -}}
{{- if ne (mod (int .Values.runConfig.model.num_layers) $pp) 0 -}}
{{- fail "model.num_layers must be divisible by pipeline_parallel_stages" -}}
{{- end -}}
{{- $config := deepCopy .Values.runConfig -}}
{{- $_ := set $config.general "n_workers" (mul $nodes $gpus) -}}
{{- $_ := set $config.general "gpu_per_node" $gpus -}}
{{- toYaml $config -}}
{{- end -}}