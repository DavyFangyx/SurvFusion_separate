# configs/presets.sh
# 定义 apply_preset <preset_name>：
#   - 设置 MODEL（传给 main.py 的 --modality）
#   - 设置 EXTRA_ARGS 数组（追加到命令末尾）
#   - 设置 RESULTS_SUBDIR（用于对齐 Python 侧真实结果目录）

apply_preset() {
    local preset="$1"

    EXTRA_ARGS=()
    RESULTS_SUBDIR=""

    case "$preset" in
        # ========== 单模态 C ==========
        mlp_clinic_mean)
            MODEL="mlp_clinic_mean"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mlp_clinic_flatten)
            MODEL="mlp_clinic_flatten"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_clinic_mean)
            MODEL="snn_clinic_mean"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_clinic_flatten)
            MODEL="snn_clinic_flatten"
            RESULTS_SUBDIR="$MODEL"
            ;;
        clinic_cox)
            MODEL="clinic_cox"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 单模态 G ==========
        mlp_gene)
            MODEL="mlp_gene"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_gene)
            MODEL="snn_gene"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mlp_gene_f)
            MODEL="mlp_gene_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        snn_gene_f)
            MODEL="snn_gene_f"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 单模态 WSI ==========
        mlp_wsi)
            MODEL="mlp_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;
        abmil_wsi)
            MODEL="abmil_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;
        transmil_wsi)
            MODEL="transmil_wsi"
            RESULTS_SUBDIR="$MODEL"
            ;;

        # ========== 多模态基线 WSI+G ==========
        porpoise)
            MODEL="porpoise"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(--fusion "$FUSION")
            ;;
        survpath)
            MODEL="survpath"
            RESULTS_SUBDIR="$MODEL"
            ;;
        mcat)
            MODEL="mcat"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(--fusion "$FUSION")
            ;;

        # ========== 主模型 / 消融 ==========
        survpgc_f)
            MODEL="survpgc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survpc_f)
            MODEL="survpc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survgc_f)
            MODEL="survgc_f"
            RESULTS_SUBDIR="$MODEL"
            ;;
        survtri_snn_concat)
            MODEL="survtri_snn_concat"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            ;;
        survtri_snn_mhsa)
            MODEL="survtri_snn_mhsa"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
            )
            ;;
        survtri_mlp_concat)
            MODEL="survtri_mlp_concat"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            ;;
        survtri_mlp_mhsa)
            MODEL="survtri_mlp_mhsa"
            RESULTS_SUBDIR="$MODEL"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${MODEL}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
            )
            ;;
        survtri_poe_vae_A)
            MODEL="survtri_poe_vae"
            POE_VARIANT="A"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_B)
            MODEL="survtri_poe_vae"
            POE_VARIANT="B"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survtri_poe_vae_C)
            MODEL="survtri_poe_vae"
            POE_VARIANT="C"
            BAG_LOSS="cox_surv"
            RESULTS_SUBDIR="${MODEL}__${POE_VARIANT}"
            if [ "$SELECTED_MODALITIES" != "wsi,gene,clinic" ]; then
                RESULTS_SUBDIR="${RESULTS_SUBDIR}__${SELECTED_MODALITIES//,/_}"
            fi
            EXTRA_ARGS=(
                --label_dim "$LABEL_DIM"
                --poe_variant "$POE_VARIANT"
                --poe_surv_lambda "$POE_SURV_LAMBDA"
                --poe_modality_dropout "$POE_MODALITY_DROPOUT"
                --poe_decoder_hidden_dim "$POE_DECODER_HIDDEN_DIM"
                --poe_mmhid "$POE_MMHID"
                --poe_beta_target "$POE_BETA_TARGET"
                --poe_transformer_layers "$POE_TRANSFORMER_LAYERS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;

        # ========== SurvFusion 变体 ==========
        survfusion_noalign)
            MODEL="survfusion_noalign"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_joint)
            MODEL="survfusion_joint"
            RESULTS_SUBDIR="$MODEL"
            EXTRA_ARGS=(
                --num_heads "$NUM_HEADS"
                --clip_lambda "$CLIP_LAMBDA"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_mhsa)
            MODEL="survfusion_separate"
            RESULTS_SUBDIR="${MODEL}_mhsa_$(printf '%g' "$CLIP_WEIGHT_IT")_$(printf '%g' "$CLIP_WEIGHT_IS")_$(printf '%g' "$CLIP_WEIGHT_TS")"
            FUSION_TYPE="mhsa"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --num_heads "$NUM_HEADS"
                --clip_weight_IT "$CLIP_WEIGHT_IT"
                --clip_weight_IS "$CLIP_WEIGHT_IS"
                --clip_weight_TS "$CLIP_WEIGHT_TS"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_concat)
            MODEL="survfusion_separate"
            FUSION_TYPE="concat"
            RESULTS_SUBDIR="${MODEL}_${FUSION_TYPE}"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;
        survfusion_separate_mean)
            MODEL="survfusion_separate"
            FUSION_TYPE="mean_concat"
            RESULTS_SUBDIR="${MODEL}_${FUSION_TYPE}"
            EXTRA_ARGS=(
                --fusion_type "$FUSION_TYPE"
                --lr_stage1 "$LR_STAGE1"
                --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
                --batch_size_stage1 "$BATCH_SIZE_STAGE1"
            )
            ;;

        *)
            echo "[presets.sh] Unknown preset: $preset" >&2
            exit 2
            ;;
    esac
}
