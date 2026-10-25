{{- define "pipeline-training.name" -}}
{{- printf "train-%s" .Release.Name | trunc 56 | trimSuffix "-" -}}
{{- end -}}

{{- define "pipeline-training.config" -}}
{{- $nodes := int .Values.numNodes -}}
{{- $gpus := int .Values.gpusPerNode -}}
{{- $dp := int .Values.runConfig.train.shard_data.DP.shard -}}
{{- $fsdp := int .Values.runConfig.train.shard_data.FSDP.shard -}}
{{- $pp := int .Values.runConfig.model.shard_model.PP.shard -}}
{{- $tp := int .Values.runConfig.model.shard_model.TP.shard -}}
{{- $ep := int .Values.runConfig.model.shard_model.EP.shard -}}
{{- $micro := int .Values.runConfig.train.microbatch_num -}}
{{- if or (lt $nodes 1) (lt $gpus 1) (lt $fsdp 1) (lt $dp 1) (lt $pp 1) (lt $tp 1) (lt $ep 1) (lt $micro 1) -}}
{{- fail "node, GPU, FSDP, DP, PP, TP, EP and microbatch counts must be positive" -}}
{{- end -}}
{{- $data := mul $fsdp $dp -}}
{{- $independentEP := $ep -}}
{{- if .Values.runConfig.model.shard_model.EP.use_tp_topology -}}
{{- $independentEP = 1 -}}
{{- end -}}
{{- if ne (mul $data $pp $tp $independentEP) (mul $nodes $gpus) -}}
{{- fail "Independent TP * EP * FSDP * PP * DP shards must equal numNodes * gpusPerNode (shared EP is excluded)" -}}
{{- end -}}
{{- if or (lt (int .Values.runConfig.train.global_batch_size) 1) (ne (mod (int .Values.runConfig.train.global_batch_size) (mul $data $micro)) 0) -}}
{{- fail "global_batch_size must be positive and divisible by FSDP * DP * microbatch_num" -}}
{{- end -}}
{{- if ne (mod (int .Values.runConfig.model.num_layers_model) $pp) 0 -}}
{{- fail "model.num_layers_model must be divisible by PP.shard" -}}
{{- end -}}
{{- $config := deepCopy .Values.runConfig -}}
{{- $_ := set $config.general "n_workers" (mul $nodes $gpus) -}}
{{- $_ := set $config.general "gpu_per_node" $gpus -}}
{{- toYaml $config -}}
{{- end -}}