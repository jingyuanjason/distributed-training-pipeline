{{- define "pipeline-training.name" -}}
{{- printf "train-%s" .Release.Name | trunc 56 | trimSuffix "-" -}}
{{- end -}}

{{- define "pipeline-training.config" -}}
{{- $nodes := int .Values.numNodes -}}
{{- $gpus := int .Values.gpusPerNode -}}
{{- $dp := int .Values.runConfig.train.data_parallel_num -}}
{{- $fsdp := $dp -}}
{{- if hasKey .Values.runConfig.train "fsdp_parallel_num" -}}
{{- $fsdp = int .Values.runConfig.train.fsdp_parallel_num -}}
{{- else -}}
{{- $dp = 1 -}}
{{- end -}}
{{- $pp := int .Values.runConfig.train.pipeline_parallel_stages -}}
{{- $micro := int .Values.runConfig.train.microbatch_num -}}
{{- if or (lt $nodes 1) (lt $gpus 1) (lt $fsdp 1) (lt $dp 1) (lt $pp 1) (lt $micro 1) -}}
{{- fail "node, GPU, FSDP, DDP, PP and microbatch counts must be positive" -}}
{{- end -}}
{{- $data := mul $fsdp $dp -}}
{{- if ne (mul $data $pp) (mul $nodes $gpus) -}}
{{- fail "fsdp_parallel_num * data_parallel_num * pipeline_parallel_stages must equal numNodes * gpusPerNode" -}}
{{- end -}}
{{- if ne (mod $gpus $data) 0 -}}
{{- fail "FSDP * DDP must divide gpusPerNode to keep each stage block node-local" -}}
{{- end -}}
{{- if or (lt (int .Values.runConfig.train.batch_size) 1) (ne (mod (int .Values.runConfig.train.batch_size) (mul $data $micro)) 0) -}}
{{- fail "batch_size must be positive and divisible by FSDP * DDP * microbatch_num" -}}
{{- end -}}
{{- if ne (mod (int .Values.runConfig.model.num_layers) $pp) 0 -}}
{{- fail "model.num_layers must be divisible by pipeline_parallel_stages" -}}
{{- end -}}
{{- $config := deepCopy .Values.runConfig -}}
{{- $_ := set $config.general "n_workers" (mul $nodes $gpus) -}}
{{- $_ := set $config.general "gpu_per_node" $gpus -}}
{{- toYaml $config -}}
{{- end -}}